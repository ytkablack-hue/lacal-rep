"""Own Triton kernel: NF4 dequantisation (with the double-quant statistics decoded in the same pass).

bitsandbytes dequantises every 4-bit weight on every forward and backward call (~390 calls per
training step here). With double quantisation (Soup's default) each call is several launches: decode
the uint8 block statistics, add the offset, then the NF4 kernel. This file does all of it in one
Triton kernel and writes the fp16 weight once.

Maths is exactly bitsandbytes':  w = nf4_table[nibble] * absmax,  where, for nested statistics,
absmax = code2[uint8] * absmax2[block // 256] + offset.  Packing: the HIGH nibble of a byte is the
first element, the LOW nibble the second.  Without double quantisation the result is bit-identical.
With it, the compiler fuses `code2 * absmax2 + offset` into one FMA, which can move the absmax by
one fp32 ulp, i.e. at most one fp16 rounding step on a rare weight element.

Two integration points (both are plain Python names looked up at call time):
- `bitsandbytes.functional.dequantize_4bit`, used by MatMul4Bit.backward;
- `bitsandbytes.backends.cuda.ops._dequant_linear_fallback`, used by the forward `gemm_4bit` op
  whenever it dequantises and calls F.linear (every batch above 1536 tokens).
"""

import importlib

import torch
import triton
import triton.language as tl

# NF4 code book (the QLoRA table). The bitsandbytes table is preferred when it can be imported.
_NF4_FALLBACK = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
]
_TABLES = {}
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
BLOCK_BYTES = 512  # bytes per program = 1024 output elements


@triton.jit
def nf4_dequant_kernel(packed_ptr, absmax_ptr, code2_ptr, absmax2_ptr, offset_ptr, table_ptr,
                       out_ptr, n_bytes, n_elems,
                       BS_HALF: tl.constexpr, BS2: tl.constexpr, NESTED: tl.constexpr,
                       BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    byte = pid * BLOCK + tl.arange(0, BLOCK)
    mask = byte < n_bytes

    packed = tl.load(packed_ptr + byte, mask=mask, other=0).to(tl.int32)
    hi = packed >> 4
    lo = packed & 0xF

    blk = byte // BS_HALF  # quantisation block of both elements of this byte
    if NESTED:
        q = tl.load(absmax_ptr + blk, mask=mask, other=0).to(tl.int32)
        code = tl.load(code2_ptr + q)
        scale2 = tl.load(absmax2_ptr + blk // BS2, mask=mask, other=0.0)
        offset = tl.load(offset_ptr)
        absmax = code * scale2 + offset
    else:
        absmax = tl.load(absmax_ptr + blk, mask=mask, other=0.0)

    v_hi = tl.load(table_ptr + hi) * absmax
    v_lo = tl.load(table_ptr + lo) * absmax
    out = tl.interleave(v_hi, v_lo)  # [hi0, lo0, hi1, lo1, ...] = elements 2b, 2b+1

    idx = pid * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
    tl.store(out_ptr + idx, out.to(out_ptr.dtype.element_ty), mask=idx < n_elems)


def _nf4_table(device):
    key = str(device)
    table = _TABLES.get(key)
    if table is None:
        try:
            from bitsandbytes.functional import get_4bit_type
            table = get_4bit_type("nf4", device=device).float().contiguous()
        except Exception:
            table = torch.tensor(_NF4_FALLBACK, dtype=torch.float32, device=device)
        _TABLES[key] = table
    return table


def dequantize_nf4_raw(packed, absmax, shape, dtype, blocksize, *, absmax2=None, code2=None,
                       offset=None, blocksize2=256):
    """Dequantise packed NF4 data. `absmax` holds the fp32 block scales, or, when `absmax2` is
    given (double quantisation), their uint8 codes, decoded with `code2`, `absmax2` and `offset`."""
    packed = packed.contiguous().view(-1)
    out = torch.empty(tuple(shape), dtype=dtype, device=packed.device)
    n_bytes, n_elems = packed.numel(), out.numel()
    table = _nf4_table(packed.device)
    nested = absmax2 is not None
    if nested:
        absmax = absmax.contiguous()
        code2 = code2.float().contiguous()
        absmax2 = absmax2.float().contiguous()
        offset = offset.float().contiguous()
    else:
        absmax = absmax.float().contiguous()
        code2 = absmax2 = offset = table  # unused placeholders
        blocksize2 = 1
    grid = (triton.cdiv(n_bytes, BLOCK_BYTES),)
    nf4_dequant_kernel[grid](packed, absmax, code2, absmax2, offset, table, out, n_bytes, n_elems,
                             BS_HALF=blocksize // 2, BS2=blocksize2, NESTED=nested,
                             BLOCK=BLOCK_BYTES)
    return out


def _supported_state(quant_state):
    blocksize = getattr(quant_state, "blocksize", None)
    if not isinstance(blocksize, int) or blocksize < 2 or blocksize % 2:
        return False
    if quant_state.dtype not in _FLOAT_DTYPES:
        return False
    if getattr(quant_state, "nested", False):
        state2 = getattr(quant_state, "state2", None)
        if state2 is None or not isinstance(getattr(state2, "blocksize", None), int):
            return False
        if getattr(state2, "code", None) is None or not torch.is_tensor(quant_state.offset):
            return False
    return True


def dequantize_nf4(packed, quant_state):
    """Dequantise a packed NF4 weight to `quant_state.shape` in `quant_state.dtype`."""
    if getattr(quant_state, "nested", False):
        state2 = quant_state.state2
        return dequantize_nf4_raw(packed, quant_state.absmax, quant_state.shape,
                                  quant_state.dtype, quant_state.blocksize, absmax2=state2.absmax,
                                  code2=state2.code, offset=quant_state.offset,
                                  blocksize2=state2.blocksize)
    return dequantize_nf4_raw(packed, quant_state.absmax, quant_state.shape, quant_state.dtype,
                              quant_state.blocksize)


def _patch_functional():
    """bitsandbytes.functional.dequantize_4bit -> our kernel for NF4 weights (backward pass)."""
    try:
        import bitsandbytes.functional as bnb_functional
        original = bnb_functional.dequantize_4bit
    except Exception:
        return
    if getattr(original, "_own_triton_nf4", False):
        return

    def dequantize_4bit(A, quant_state=None, absmax=None, out=None, blocksize=None,
                        quant_type="fp4"):
        if (quant_state is not None and out is None and absmax is None
                and getattr(quant_state, "quant_type", None) == "nf4"
                and getattr(A, "is_cuda", False) and _supported_state(quant_state)):
            result = dequantize_nf4(A, quant_state)
            # same shim as bitsandbytes: a transposed packed weight returns the transposed result
            return result.t() if A.shape[0] == 1 else result
        return original(A, quant_state, absmax, out, blocksize, quant_type)

    dequantize_4bit._own_triton_nf4 = True
    dequantize_4bit._own_original = original
    bnb_functional.dequantize_4bit = dequantize_4bit


def _patch_gemm_fallback():
    """The CUDA `gemm_4bit` op (forward pass) dequantises through `_dequant_linear_fallback` and
    then calls F.linear. Same GEMM, our dequantisation."""
    try:
        import bitsandbytes  # noqa: F401  (registers the backends)
        ops = importlib.import_module("bitsandbytes.backends.cuda.ops")
        original = ops._dequant_linear_fallback
    except Exception:
        return
    if getattr(original, "_own_triton_nf4", False):
        return

    def _dequant_linear_fallback(A, B, shapeB, absmax, blocksize, quant_type, bias=None,
                                 absmax_8bit=None, absmax_code=None, absmax_offset=None):
        nested = absmax_8bit is not None
        if (quant_type == "nf4" and getattr(A, "is_cuda", False) and A.dtype in _FLOAT_DTYPES
                and isinstance(blocksize, int) and blocksize >= 2 and blocksize % 2 == 0
                and (not nested or (absmax_code is not None and torch.is_tensor(absmax_offset)))):
            if nested:
                weight = dequantize_nf4_raw(B, absmax_8bit, shapeB, A.dtype, blocksize,
                                            absmax2=absmax, code2=absmax_code,
                                            offset=absmax_offset)
            else:
                weight = dequantize_nf4_raw(B, absmax, shapeB, A.dtype, blocksize)
            return torch.nn.functional.linear(A, weight, bias)
        return original(A, B, shapeB, absmax, blocksize, quant_type, bias, absmax_8bit,
                        absmax_code, absmax_offset)

    _dequant_linear_fallback._own_triton_nf4 = True
    _dequant_linear_fallback._own_original = original
    ops._dequant_linear_fallback = _dequant_linear_fallback


def patch_bnb_dequantize():
    """Use our NF4 kernel for every bitsandbytes dequantisation (forward and backward). Anything
    unusual falls back to the original code. Safe if bitsandbytes is a stub."""
    _patch_functional()
    _patch_gemm_fallback()

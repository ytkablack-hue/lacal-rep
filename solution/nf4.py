"""Own Triton kernel: NF4 dequantisation (with the double-quant statistics decoded in the same pass).

bitsandbytes dequantises every 4-bit weight on every forward and backward call (~390 calls per
training step here). With double quantisation (Soup's default) each call is several launches: decode
the uint8 block statistics, add the offset, then the NF4 kernel. This file does all of it in one
Triton kernel and writes the fp16 weight once.

Maths is exactly bitsandbytes':  w = nf4_table[nibble] * absmax,  where, for nested statistics,
absmax = code2[uint8] * absmax2[block // 256] + offset.  Packing: the HIGH nibble of a byte is the
first element, the LOW nibble the second.  The only possible difference to bitsandbytes is a fused
multiply-add inside the nested absmax, i.e. at most one fp16 rounding step on a rare element.
"""

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


def _supported(quant_state):
    blocksize = getattr(quant_state, "blocksize", None)
    if not isinstance(blocksize, int) or blocksize < 2 or blocksize % 2:
        return False
    if quant_state.dtype not in (torch.float16, torch.bfloat16, torch.float32):
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
    packed = packed.contiguous().view(-1)
    n_bytes = packed.numel()
    out = torch.empty(quant_state.shape, dtype=quant_state.dtype, device=packed.device)
    n_elems = out.numel()
    table = _nf4_table(packed.device)
    nested = bool(getattr(quant_state, "nested", False))
    if nested:
        state2 = quant_state.state2
        absmax = quant_state.absmax.contiguous()
        code2 = state2.code.float().contiguous()
        absmax2 = state2.absmax.float().contiguous()
        offset = quant_state.offset.float().contiguous()
        bs2 = state2.blocksize
    else:
        absmax = quant_state.absmax.float().contiguous()
        code2 = absmax2 = offset = table  # unused placeholders
        bs2 = 1
    grid = (triton.cdiv(n_bytes, BLOCK_BYTES),)
    nf4_dequant_kernel[grid](packed, absmax, code2, absmax2, offset, table, out, n_bytes, n_elems,
                             BS_HALF=quant_state.blocksize // 2, BS2=bs2, NESTED=nested,
                             BLOCK=BLOCK_BYTES)
    return out


def patch_bnb_dequantize():
    """Route bitsandbytes' `dequantize_4bit` (used by MatMul4Bit.backward) through our kernel for
    NF4 weights. Anything unusual falls back to the original. Safe if bitsandbytes is a stub."""
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
                and getattr(A, "is_cuda", False) and _supported(quant_state)):
            result = dequantize_nf4(A, quant_state)
            # same shim as bitsandbytes: a transposed packed weight returns the transposed result
            return result.t() if A.shape[0] == 1 else result
        return original(A, quant_state, absmax, out, blocksize, quant_type)

    dequantize_4bit._own_triton_nf4 = True
    dequantize_4bit._own_original = original
    bnb_functional.dequantize_4bit = dequantize_4bit

"""Our NF4 dequantisation against bitsandbytes' own, with and without double quantisation.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("double_quant", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(1536, 1536), (8960, 1536), (257, 130)])
def test_dequantize_nf4_matches_bitsandbytes(double_quant, dtype, shape):
    F = pytest.importorskip("bitsandbytes.functional")
    from solution.nf4 import dequantize_nf4

    torch.manual_seed(0)
    weight = torch.randn(shape, device="cuda", dtype=dtype)
    packed, state = F.quantize_4bit(weight, blocksize=64, compress_statistics=double_quant,
                                    quant_type="nf4")
    original = getattr(F.dequantize_4bit, "_own_original", F.dequantize_4bit)
    ref = original(packed, state)
    out = dequantize_nf4(packed, state)

    assert out.shape == ref.shape and out.dtype == ref.dtype
    mismatched = (out != ref).float().mean().item()
    if not double_quant:
        # one multiplication, no fusing possible: bit-identical to bitsandbytes
        assert mismatched == 0.0, f"{mismatched:.2e} of the elements differ from bitsandbytes"
    elif dtype == torch.float16:
        # the production dtype: a fused multiply-add can move one fp16 rounding step, rarely
        torch.testing.assert_close(out.float(), ref.float(), rtol=2e-3, atol=1e-6)
        assert mismatched < 1e-3, f"{mismatched:.2e} of the elements differ from bitsandbytes"
    else:
        # fp32 output shows the fused multiply-add as a last-bit difference; a wrong table or
        # packing order would be off by orders of magnitude more
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("double_quant", [True, False])
def test_patched_gemm_4bit_matches_original(double_quant):
    """The forward op: the same call with bitsandbytes' own dequantisation and with ours."""
    F = pytest.importorskip("bitsandbytes.functional")
    ops = pytest.importorskip("bitsandbytes.backends.cuda.ops")
    from solution import nf4

    torch.manual_seed(0)
    n, k = 1536, 1536
    weight = torch.randn(n, k, device="cuda", dtype=torch.float16)
    packed, state = F.quantize_4bit(weight, blocksize=64, compress_statistics=double_quant,
                                    quant_type="nf4")
    x = torch.randn(4, 512, k, device="cuda", dtype=torch.float16)  # 2048 tokens: dequant + linear

    def call():
        extra = {}
        absmax = state.absmax
        if state.nested:
            absmax = state.state2.absmax
            extra = dict(absmax_8bit=state.absmax, absmax_code=state.state2.code,
                         absmax_offset=state.offset)
        return torch.ops.bitsandbytes.gemm_4bit.default(
            x, packed, state.shape, absmax, state.blocksize, state.quant_type, bias=None, **extra)

    current = ops._dequant_linear_fallback
    ops._dequant_linear_fallback = getattr(current, "_own_original", current)
    ref = call()
    nf4.patch_bnb_dequantize()
    assert getattr(ops._dequant_linear_fallback, "_own_triton_nf4", False)
    out = call()

    assert out.shape == ref.shape and out.dtype == ref.dtype
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-3, atol=1e-2)

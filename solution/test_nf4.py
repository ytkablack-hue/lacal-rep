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
    # a fused multiply-add may move one fp16 rounding step on a rare element: allow 1 ulp
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-3, atol=1e-6)
    mismatched = (out != ref).float().mean().item()
    assert mismatched < 1e-3, f"{mismatched:.2e} of the elements differ from bitsandbytes"

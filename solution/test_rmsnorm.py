"""Correctness of the fused RMSNorm against plain fp32 PyTorch, forward and backward.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}


def reference(x, w, eps):
    xf = x.float()
    xn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w.float() * xn


@pytest.mark.parametrize("dtypes", [(torch.float16, torch.float16), (torch.float32, torch.float32),
                                    (torch.float16, torch.float32)])
@pytest.mark.parametrize("shape", [(4, 512, 1536), (3, 77, 1001)])
@pytest.mark.parametrize("weight_grad", [False, True])
def test_rmsnorm_matches_pytorch(dtypes, shape, weight_grad):
    from solution.rmsnorm import rmsnorm

    x_dtype, w_dtype = dtypes
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=x_dtype, requires_grad=True)
    w = (torch.randn(shape[-1], device="cuda") * 0.5 + 1.0).to(w_dtype).requires_grad_(weight_grad)
    out_dtype = torch.promote_types(x_dtype, w_dtype)
    grad = torch.randn(shape, device="cuda", dtype=out_dtype)

    xr = x.detach().float().requires_grad_(True)
    wr = w.detach().float().requires_grad_(weight_grad)
    ref = reference(xr, wr, 1e-6)
    ref.backward(grad.float())

    out = rmsnorm(x, w, 1e-6)
    out.backward(grad)

    assert out.dtype == out_dtype
    tol = TOLERANCE[x_dtype]
    torch.testing.assert_close(out.float(), ref.detach(), rtol=tol, atol=tol)
    torch.testing.assert_close(x.grad.float(), xr.grad, rtol=tol, atol=tol)
    if weight_grad:
        wtol = 5e-2 if w_dtype == torch.float16 else 1e-3
        torch.testing.assert_close(w.grad.float(), wr.grad, rtol=wtol, atol=wtol)
    else:
        assert w.grad is None

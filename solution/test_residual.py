"""Residual add + RMSNorm in one kernel against the separate stock-style ops, forward and backward.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}


@pytest.mark.parametrize("dtypes", [(torch.float32, torch.float16, torch.float32),
                                    (torch.float32, torch.float32, torch.float32),
                                    (torch.float16, torch.float16, torch.float16)])
@pytest.mark.parametrize("shape", [(4, 512, 1536), (3, 77, 1001)])
def test_add_rmsnorm_matches_separate_ops(dtypes, shape):
    from solution.residual import add_rmsnorm
    from solution.rmsnorm import rmsnorm

    r_dtype, a_dtype, w_dtype = dtypes
    s_dtype = torch.promote_types(r_dtype, a_dtype)
    y_dtype = torch.promote_types(s_dtype, w_dtype)
    torch.manual_seed(0)
    r0 = torch.randn(shape, device="cuda", dtype=r_dtype)
    a0 = torch.randn(shape, device="cuda", dtype=a_dtype)
    w = (torch.randn(shape[-1], device="cuda") * 0.5 + 1.0).to(w_dtype)
    ds = torch.randn(shape, device="cuda", dtype=s_dtype)
    dy = torch.randn(shape, device="cuda", dtype=y_dtype)

    r_ref = r0.clone().requires_grad_(True)
    a_ref = a0.clone().requires_grad_(True)
    s_ref = r_ref + a_ref
    y_ref = rmsnorm(s_ref, w, 1e-6)
    torch.autograd.backward([s_ref, y_ref], [ds, dy])

    r = r0.clone().requires_grad_(True)
    a = a0.clone().requires_grad_(True)
    s, y = add_rmsnorm(r, a, w, 1e-6)
    torch.autograd.backward([s, y], [ds, dy])

    assert s.dtype == s_ref.dtype and y.dtype == y_ref.dtype
    assert r.grad.dtype == r_dtype and a.grad.dtype == a_dtype
    tol = TOLERANCE[s_dtype]
    torch.testing.assert_close(s.float(), s_ref.detach().float(), rtol=tol, atol=tol)
    torch.testing.assert_close(y.float(), y_ref.detach().float(), rtol=tol, atol=tol)
    # each gradient is compared at the precision of its own dtype: a last-bit difference before
    # the final rounding may flip one fp16 rounding step in a few elements
    r_tol, a_tol = TOLERANCE[r_dtype], TOLERANCE[a_dtype]
    torch.testing.assert_close(r.grad.float(), r_ref.grad.float(), rtol=r_tol, atol=r_tol)
    torch.testing.assert_close(a.grad.float(), a_ref.grad.float(), rtol=a_tol, atol=a_tol)
    if a_dtype == torch.float16:
        assert (a.grad != a_ref.grad).float().mean().item() < 1e-3

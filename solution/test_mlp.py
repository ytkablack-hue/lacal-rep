"""The half-precision MLP against the stock path, forward and backward.

Run on a GPU: `python -m pytest solution -q`.
"""

import types

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_swiglu_half_matches_example(shape):
    """Same maths as the starter example's fp32 kernel, followed by the fp16 rounding that the
    stock path performs in the next linear's input cast."""
    from solution.mlp import swiglu_half
    from solution.swiglu import swiglu

    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=torch.float16)

    ref = swiglu(gate.float(), up.float()).half()
    ref.backward(grad)
    ref_dgate, ref_dup = gate.grad.clone(), up.grad.clone()
    gate.grad = up.grad = None

    out = swiglu_half(gate, up)
    out.backward(grad)

    assert out.dtype == torch.float16 and gate.grad.dtype == torch.float16
    for ours, theirs in ((out, ref), (gate.grad, ref_dgate), (up.grad, ref_dup)):
        torch.testing.assert_close(ours.float(), theirs.float(), rtol=2e-3, atol=2e-3)
        assert (ours != theirs).float().mean().item() < 1e-3


def _make_mlp(hidden, inter, compress):
    bnb = pytest.importorskip("bitsandbytes")

    def linear(n_in, n_out):
        layer = bnb.nn.Linear4bit(n_in, n_out, bias=False, compute_dtype=torch.float16,
                                  compress_statistics=compress, quant_type="nf4")
        weight = torch.randn(n_out, n_in, dtype=torch.float16) * 0.05
        layer.weight = bnb.nn.Params4bit(weight, requires_grad=False, quant_type="nf4",
                                         compress_statistics=compress)
        return layer.cuda()

    class Mlp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(hidden_act="silu")
            self.gate_proj = linear(hidden, inter)
            self.up_proj = linear(hidden, inter)
            self.down_proj = linear(inter, hidden)

    return Mlp()


@pytest.mark.parametrize("compress", [True, False])
@pytest.mark.parametrize("tokens", [(2, 100), (4, 512)])
def test_half_mlp_matches_stock_path(compress, tokens):
    """Stock path: down(silu(gate(x)) * up(x)) with an fp32 hidden state and fp16 autocast, as in
    training. Ours: the half-precision path. Same modules, same input, same output gradient."""
    from solution.mlp import half_forward, is_eligible

    torch.manual_seed(0)
    mlp = _make_mlp(1536, 8960, compress)
    x = torch.randn(*tokens, 1536, device="cuda", dtype=torch.float32, requires_grad=True)
    grad = torch.randn(*tokens, 1536, device="cuda", dtype=torch.float32)
    assert is_eligible(mlp, x)

    with torch.autocast("cuda", dtype=torch.float16):
        ref = mlp.down_proj(torch.nn.functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
    ref.backward(grad)
    ref_dx = x.grad.clone()
    x.grad = None

    with torch.autocast("cuda", dtype=torch.float16):
        out = half_forward(mlp, x)
    out.backward(grad)

    assert out.dtype == ref.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-2)
    torch.testing.assert_close(x.grad, ref_dx, rtol=5e-3, atol=5e-2)

"""The fused residual add + RMSNorm against the separate ops, and the patched decoder layer against
the stock one.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-5}


# (residual, branch, weight): the real model has an fp32 residual stream and an fp16 attention output
@pytest.mark.parametrize("dtypes", [(torch.float32, torch.float32, torch.float32),
                                    (torch.float32, torch.float16, torch.float32),
                                    (torch.float16, torch.float16, torch.float32),
                                    (torch.float16, torch.float16, torch.float16)])
@pytest.mark.parametrize("shape", [(4, 300, 1536), (3, 77, 1001)])
@pytest.mark.parametrize("sum_used", [True, False])
def test_add_rmsnorm_matches_separate_ops(dtypes, shape, sum_used):
    from solution.residual import add_rmsnorm
    from solution.rmsnorm import rmsnorm

    x_dtype, b_dtype, w_dtype = dtypes
    s_dtype = torch.promote_types(x_dtype, b_dtype)
    out_dtype = torch.promote_types(s_dtype, w_dtype)
    torch.manual_seed(0)
    residual = torch.randn(shape, device="cuda", dtype=x_dtype)
    branch = torch.randn(shape, device="cuda", dtype=b_dtype)
    weight = (torch.randn(shape[-1], device="cuda") * 0.5 + 1.0).to(w_dtype)
    grad_y = torch.randn(shape, device="cuda", dtype=out_dtype)
    grad_s = torch.randn(shape, device="cuda", dtype=s_dtype)

    def run(fused):
        r = residual.clone().requires_grad_(True)
        b = branch.clone().requires_grad_(True)
        if fused:
            s, y = add_rmsnorm(r, b, weight, 1e-6)
        else:
            s = r + b
            y = rmsnorm(s, weight, 1e-6)
        outputs, grads = ([y, s], [grad_y, grad_s]) if sum_used else ([y], [grad_y])
        torch.autograd.backward(outputs, grads)
        return s.detach(), y.detach(), r.grad, b.grad

    ref = run(fused=False)
    out = run(fused=True)

    assert out[0].dtype == ref[0].dtype == s_dtype and out[1].dtype == ref[1].dtype == out_dtype
    assert out[2].dtype == ref[2].dtype == x_dtype and out[3].dtype == ref[3].dtype == b_dtype
    assert torch.equal(out[0], ref[0])  # the sum is the same fp32 / fp16 addition
    tol = max(TOLERANCE[x_dtype], TOLERANCE[b_dtype])
    for ours, theirs in zip(out[1:], ref[1:]):
        torch.testing.assert_close(ours.float(), theirs.float(), rtol=tol, atol=tol)


def test_patched_decoder_layer_matches_stock():
    from transformers.models.qwen2 import modeling_qwen2 as modeling
    from solution.residual import patch_qwen2_decoder_residual

    config = modeling.Qwen2Config(vocab_size=200, hidden_size=64, intermediate_size=128,
                                  num_hidden_layers=2, num_attention_heads=4,
                                  num_key_value_heads=2, max_position_embeddings=128)
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = modeling.Qwen2Model(config).cuda().train()
    ids = torch.randint(0, 200, (3, 40), device="cuda")

    cls = modeling.Qwen2DecoderLayer
    patch_qwen2_decoder_residual()
    assert getattr(cls, "_own_residual", False), (
        "the patch did not apply: Qwen2DecoderLayer.forward has another layout in this "
        "transformers version (print inspect.getsource(Qwen2DecoderLayer.forward))")
    patched, original = cls.forward, cls._own_residual_original

    def run(forward):
        cls.forward = forward
        try:
            model.zero_grad(set_to_none=True)
            out = model(input_ids=ids, use_cache=False).last_hidden_state
            (out ** 2).sum().backward()
            return out.detach().clone(), model.embed_tokens.weight.grad.clone()
        finally:
            cls.forward = patched

    ref_out, ref_grad = run(original)
    out, grad = run(patched)
    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(grad, ref_grad, rtol=1e-3, atol=1e-4)

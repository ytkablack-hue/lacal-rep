"""Packed forward (no padding) against the stock padded forward, and the LoRA dropout wrapper.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _tiny_model():
    from transformers.models.qwen2 import modeling_qwen2 as m

    config = m.Qwen2Config(vocab_size=200, hidden_size=64, intermediate_size=128,
                           num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                           max_position_embeddings=128)
    config._attn_implementation = "sdpa"
    return m, m.Qwen2Model(config).cuda().train()


def _batch(lengths, vocab=200):
    length = max(lengths)
    ids = torch.randint(0, vocab, (len(lengths), length), device="cuda")
    mask = torch.zeros(len(lengths), length, dtype=torch.long, device="cuda")
    for row, n in enumerate(lengths):
        mask[row, :n] = 1
    return ids, mask


@pytest.mark.parametrize("lengths", [[20, 11, 5, 17], [9, 9, 3, 1], [13, 13, 13, 13]])
def test_packed_forward_matches_padded(lengths):
    from solution.unpad import patch_qwen2_unpad

    torch.manual_seed(0)
    modeling, model = _tiny_model()
    patch_qwen2_unpad()
    original = modeling.Qwen2Model._own_unpad_original
    ids, mask = _batch(lengths)
    keep = mask.bool()

    ref = original(model, input_ids=ids, attention_mask=mask).last_hidden_state
    (ref[keep] ** 2).sum().backward()
    ref_grad = model.embed_tokens.weight.grad.clone()
    model.zero_grad(set_to_none=True)

    out = model(input_ids=ids, attention_mask=mask).last_hidden_state
    assert out.shape == ref.shape and out.dtype == ref.dtype
    torch.testing.assert_close(out[keep], ref[keep], rtol=1e-4, atol=1e-4)
    if (~keep).any():
        assert (out[~keep] == 0).all()
    (out[keep] ** 2).sum().backward()
    torch.testing.assert_close(model.embed_tokens.weight.grad, ref_grad, rtol=1e-3, atol=1e-4)


def test_eval_mode_takes_stock_path():
    from solution.unpad import patch_qwen2_unpad

    torch.manual_seed(0)
    modeling, model = _tiny_model()
    patch_qwen2_unpad()
    model.eval()
    ids, mask = _batch([8, 5])
    ref = modeling.Qwen2Model._own_unpad_original(model, input_ids=ids, attention_mask=mask)
    out = model(input_ids=ids, attention_mask=mask)
    assert torch.equal(out.last_hidden_state, ref.last_hidden_state)


@pytest.mark.parametrize("shape", [(4, 37, 64), (3, 128, 1536)])
def test_packed_dropout_matches_padded_dropout(shape):
    from solution import unpad

    batch, length, hidden = shape
    torch.manual_seed(1)
    x_pad = torch.randn(shape, device="cuda", requires_grad=True)
    lengths = [length, length // 2, 3, length // 3][:batch]
    keep = torch.zeros(batch, length, dtype=torch.bool, device="cuda")
    for row, n in enumerate(lengths):
        keep[row, :n] = True
    idx = keep.reshape(-1).nonzero().squeeze(1)
    grad = torch.randn(len(idx), hidden, device="cuda")

    drop = torch.nn.Dropout(0.05).cuda().train()
    torch.manual_seed(5)
    ref = drop(x_pad)
    ref_next = torch.rand(4, device="cuda")
    ref.reshape(-1, hidden)[idx].backward(grad)
    ref_grad = x_pad.grad.reshape(-1, hidden)[idx].clone()
    x_pad.grad = None

    packed = unpad.PackedDropout(drop)
    x = x_pad.detach().reshape(-1, hidden)[idx].unsqueeze(0).clone().requires_grad_(True)
    unpad._PACK = unpad._Pack(batch, length, idx)
    try:
        torch.manual_seed(5)
        out = packed(x)
        out_next = torch.rand(4, device="cuda")
        out[0].backward(grad)
    finally:
        unpad._PACK = None

    assert torch.equal(out[0], ref.detach().reshape(-1, hidden)[idx])  # same masks, same values
    assert torch.equal(out_next, ref_next)  # the generator state afterwards is the same
    torch.testing.assert_close(x.grad[0], ref_grad)

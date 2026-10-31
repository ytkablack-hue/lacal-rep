"""Remove padding inside a batch: the decoder layers run only on real tokens.

Stock path. The collator pads the 4 rows of a step to the longest one, so on this task about 42% of
the tokens the GPU processes are padding (52,890 real vs 91,856 padded over the 64 steps). Every
GEMM, norm, MLP and LoRA layer pays for them, and so does every activation saved for backward.

Here Qwen2Model.forward packs the real tokens of the batch into one row [1, T], runs the original
forward on it, and scatters the final hidden states back to [B, L, H] (zeros at padding), so
everything after the model (lm_head, loss, TRL metrics, Soup's rewind) sees the same shapes as before.

What keeps the numbers the same as the stock path:
  * position ids: the value the stock model uses for each real token (its column in the padded row);
  * attention: a block-diagonal causal mask (token i sees token j of the SAME row with j <= i), the
    same set of keys every real query sees in the padded batch with its padding mask;
  * LoRA dropout: PEFT draws the mask for the padded [B, L, H] input. A packed input has another
    shape, which would change the random stream, so each LoRA dropout is replaced by a wrapper that
    draws the mask for the padded shape (on ones, same dtype, same generator calls) and takes the
    rows of real tokens. Masks of real tokens and the generator state stay as in the stock path.

Anything unusual (eval mode, no padding, cache, explicit position ids, hidden states requested)
takes the stock code. Safe if transformers or peft are stubs.
"""

import torch

_PACK = None  # set while a packed forward runs


class _Pack:
    def __init__(self, batch, length, idx):
        self.batch = batch
        self.length = length
        self.idx = idx
        self.total = int(idx.numel())


class _MaskScale(torch.autograd.Function):
    """x * mask * scale with the boolean mask saved (1 byte per element, like native dropout)."""

    @staticmethod
    def forward(ctx, x, mask, scale):
        ctx.save_for_backward(mask)
        ctx.scale = scale
        return x * mask * scale

    @staticmethod
    def backward(ctx, grad):
        (mask,) = ctx.saved_tensors
        return grad * mask * ctx.scale, None, None


class PackedDropout(torch.nn.Module):
    """Drop-in for a LoRA nn.Dropout: identical to it outside a packed forward; inside one, the mask
    of the padded input is drawn and only the rows of real tokens are used."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        pack = _PACK
        inner = self.inner
        if (pack is None or not inner.training or x.dim() != 3 or x.shape[0] != 1
                or x.shape[1] != pack.total or not x.is_cuda):
            return inner(x)
        hidden = x.shape[-1]
        ones = torch.ones(pack.batch, pack.length, hidden, dtype=x.dtype, device=x.device)
        _, mask = torch.native_dropout(ones, inner.p, True)  # the call F.dropout makes on CUDA
        rows = mask.view(pack.batch * pack.length, hidden).index_select(0, pack.idx)
        return _MaskScale.apply(x, rows.unsqueeze(0), 1.0 / (1.0 - inner.p))


def swap_lora_dropout(root):
    """Wrap every nn.Dropout (p > 0) inside the LoRA layers of `root`. Returns how many."""
    count = 0
    for module in list(root.modules()):
        drops = getattr(module, "lora_dropout", None)
        if isinstance(drops, torch.nn.ModuleDict):
            for key in list(drops.keys()):
                layer = drops[key]
                if isinstance(layer, torch.nn.Dropout) and layer.p > 0:
                    drops[key] = PackedDropout(layer)
                    count += 1
    return count


def _eligible(model, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds,
              use_cache, kwargs):
    return (model.training and torch.is_grad_enabled()
            and torch.is_tensor(input_ids) and input_ids.dim() == 2 and input_ids.is_cuda
            and torch.is_tensor(attention_mask) and attention_mask.shape == input_ids.shape
            and not attention_mask.is_floating_point()
            and position_ids is None and past_key_values is None and inputs_embeds is None
            and not use_cache
            and not kwargs.get("output_hidden_states") and not kwargs.get("output_attentions")
            and kwargs.get("return_dict", True) is not False)


def patch_qwen2_unpad():
    """Qwen2Model.forward on packed real tokens. Safe if transformers is a stub."""
    try:
        from transformers.models.qwen2 import modeling_qwen2
        cls = modeling_qwen2.Qwen2Model
        original = cls.forward
    except Exception:
        return
    if getattr(cls, "_own_unpad", False):
        return

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
        global _PACK  # assigned below: without this it is a local name (UnboundLocalError)

        def stock():
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)

        if not _eligible(self, input_ids, attention_mask, position_ids, past_key_values,
                         inputs_embeds, use_cache, kwargs):
            return stock()
        batch, length = input_ids.shape
        idx = attention_mask.reshape(-1).ne(0).nonzero().squeeze(1)
        total = int(idx.numel())  # one host sync per step
        if total == 0 or total == batch * length:
            return stock()

        rows = idx // length
        cols = idx % length
        arange = torch.arange(total, device=idx.device)
        mask = ((rows[:, None] == rows[None, :]) & (arange[:, None] >= arange[None, :]))
        mask = mask[None, None]
        layer_types = set(getattr(self.config, "layer_types", None) or ["full_attention"])
        mask_mapping = {t: mask for t in layer_types}

        if not getattr(self, "_own_dropout_swapped", False):
            swap_lora_dropout(self)
            self._own_dropout_swapped = True

        previous, _PACK = _PACK, _Pack(batch, length, idx)
        try:
            out = original(self, input_ids=input_ids.reshape(-1)[idx].unsqueeze(0),
                           attention_mask=mask_mapping, position_ids=cols.unsqueeze(0),
                           use_cache=False, **kwargs)
        finally:
            _PACK = previous

        is_tuple = isinstance(out, tuple)
        hidden = out[0] if is_tuple else out.last_hidden_state
        hidden = hidden.reshape(total, -1)
        full = torch.zeros(batch * length, hidden.shape[-1], dtype=hidden.dtype,
                           device=hidden.device).index_copy(0, idx, hidden)
        full = full.view(batch, length, -1)
        if is_tuple:
            return (full,) + tuple(out[1:])
        out.last_hidden_state = full
        return out

    cls._own_unpad_original = original
    cls.forward = forward
    cls._own_unpad = True

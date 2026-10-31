"""Own Triton kernels: chunked lm_head + online softmax CE, entropy and argmax.

Stock path materialises logits of shape [tokens, vocab] (151,936) in fp32, then softmax /
cross-entropy / entropy / argmax. That tensor is ~1.2 GiB at the padded batch size and is the
main reason the T4 is nearly full. TRL also walks the same logits again for entropy and
accuracy; Soup's rewind reads per-token NLL from them.

Here the lm_head GEMM is done in vocab chunks. One Triton program per chunk updates running
log-sum-exp, the label logit, argmax, and the weighted sum needed for Shannon entropy, then the
chunk is discarded. Backward recomputes the same chunks to form d(hidden) without storing
d(logits). Numbers follow the stock path: fp32 reductions, mean CE over non-ignored shifted
labels, entropy over the attention mask (prompt tokens included, padding excluded).

Ideas: online softmax (Milakov & Gimelshein), chunked / cut cross-entropy (Liger, Unsloth,
Apple CCE) — the kernels here are our own.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

IGNORE = -100
CHUNK = 4096
BLOCK_N = 64
BLOCK_C = 128

# Filled during a fused forward; TRL / rewind read it instead of a full logits tensor.
STATE = {
    "nll_shift": None,   # [B, L-1] CE none, ignore_index left as 0 and masked by labels
    "valid_shift": None,  # [B, L-1] bool
    "entropy": None,     # [B, L]
    "pred_shift": None,  # [B, L-1] argmax of logits[:, :-1]
    "mean_entropy": None,
    "mean_accuracy": None,
    "vocab": None,
}


def clear_state():
    for key in list(STATE):
        STATE[key] = None


@triton.jit
def ce_stats_kernel(logits_ptr, m_ptr, s_ptr, acc_ptr, n_rows, n_cols,
                    BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr):
    """Update online softmax stats for one vocab chunk: max, sumexp, sum(exp*x)."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_rows

    m = tl.load(m_ptr + offs_n, mask=mask_n, other=-float("inf"))
    s = tl.load(s_ptr + offs_n, mask=mask_n, other=0.0)
    acc = tl.load(acc_ptr + offs_n, mask=mask_n, other=0.0)

    for c0 in range(0, n_cols, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_c = offs_c < n_cols
        mask = mask_n[:, None] & mask_c[None, :]
        vals = tl.load(logits_ptr + offs_n[:, None] * n_cols + offs_c[None, :],
                       mask=mask, other=-float("inf")).to(tl.float32)
        tile_max = tl.max(vals, axis=1)
        m_new = tl.maximum(m, tile_max)
        scale = tl.exp(m - m_new)
        expv = tl.exp(vals - m_new[:, None])
        expv = tl.where(mask, expv, 0.0)
        vals_z = tl.where(mask, vals, 0.0)
        s = s * scale + tl.sum(expv, axis=1)
        acc = acc * scale + tl.sum(vals_z * expv, axis=1)
        m = m_new

    tl.store(m_ptr + offs_n, m, mask=mask_n)
    tl.store(s_ptr + offs_n, s, mask=mask_n)
    tl.store(acc_ptr + offs_n, acc, mask=mask_n)


@triton.jit
def ce_bwd_kernel(logits_ptr, labels_ptr, lse_ptr, dlogits_ptr, n_rows, n_cols, vocab_start,
                  ignore_index, scale, BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr):
    """dlogits = scale * (softmax - one_hot(label)) for one vocab chunk."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_rows
    labels = tl.load(labels_ptr + offs_n, mask=mask_n, other=ignore_index)
    lse = tl.load(lse_ptr + offs_n, mask=mask_n, other=0.0)
    valid = mask_n & (labels != ignore_index)

    for c0 in range(0, n_cols, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_c = offs_c < n_cols
        mask = mask_n[:, None] & mask_c[None, :]
        vals = tl.load(logits_ptr + offs_n[:, None] * n_cols + offs_c[None, :],
                       mask=mask, other=0.0).to(tl.float32)
        prob = tl.exp(vals - lse[:, None])
        local = labels - vocab_start
        onehot = (offs_c[None, :] == local[:, None]) & valid[:, None] & mask_c[None, :]
        grad = tl.where(valid[:, None] & mask, scale * (prob - onehot.to(tl.float32)), 0.0)
        tl.store(dlogits_ptr + offs_n[:, None] * n_cols + offs_c[None, :],
                 grad.to(dlogits_ptr.dtype.element_ty), mask=mask)


def _stats_launch(logits, m, s, acc):
    n_rows, n_cols = logits.shape
    grid = (triton.cdiv(n_rows, BLOCK_N),)
    ce_stats_kernel[grid](logits, m, s, acc, n_rows, n_cols, BLOCK_N=BLOCK_N, BLOCK_C=BLOCK_C)


def _bwd_launch(logits, labels, lse, scale, vocab_start):
    n_rows, n_cols = logits.shape
    dlogits = torch.empty_like(logits, dtype=torch.float32)
    grid = (triton.cdiv(n_rows, BLOCK_N),)
    ce_bwd_kernel[grid](logits, labels, lse, dlogits, n_rows, n_cols, vocab_start, IGNORE,
                        scale, BLOCK_N=BLOCK_N, BLOCK_C=BLOCK_C)
    return dlogits


def _linear_chunk(hidden, weight, start, stop):
    return F.linear(hidden, weight[start:stop])


class ChunkedLinearCE(torch.autograd.Function):
    """Mean CE of linear(hidden, weight) vs labels, without storing [N, V] logits."""

    @staticmethod
    def forward(ctx, hidden, weight, labels):
        hidden = hidden.contiguous()
        weight = weight.contiguous()
        labels = labels.reshape(-1).contiguous()
        n_rows, hidden_size = hidden.shape
        vocab = weight.shape[0]
        device = hidden.device

        m = torch.full((n_rows,), -float("inf"), device=device, dtype=torch.float32)
        s = torch.zeros(n_rows, device=device, dtype=torch.float32)
        acc = torch.zeros(n_rows, device=device, dtype=torch.float32)
        target = torch.zeros(n_rows, device=device, dtype=torch.float32)
        amax = torch.full((n_rows,), -float("inf"), device=device, dtype=torch.float32)
        aidx = torch.zeros(n_rows, device=device, dtype=torch.long)

        for start in range(0, vocab, CHUNK):
            stop = min(start + CHUNK, vocab)
            logits = _linear_chunk(hidden, weight, start, stop).float()
            _stats_launch(logits, m, s, acc)
            chunk_max, chunk_idx = logits.max(dim=-1)
            better = chunk_max > amax
            aidx = torch.where(better, chunk_idx + start, aidx)
            amax = torch.maximum(amax, chunk_max)
            hit = (labels >= start) & (labels < stop)
            if hit.any():
                target = torch.where(hit, logits[torch.arange(n_rows, device=device),
                                                 (labels - start).clamp_min(0)], target)

        lse = m + torch.log(s.clamp_min(1e-30))
        valid = labels != IGNORE
        nll = torch.where(valid, lse - target, torch.zeros_like(lse))
        n_valid = valid.sum()
        loss = nll.sum() / n_valid.clamp_min(1)
        entropy = torch.where(s > 0, lse - acc / s.clamp_min(1e-30), torch.zeros_like(lse))

        ctx.save_for_backward(hidden, weight, labels, lse)
        ctx.mark_non_differentiable(nll, entropy, aidx)
        ctx.n_valid = int(n_valid.item())
        ctx.vocab = vocab
        return loss, nll, entropy, aidx

    @staticmethod
    def backward(ctx, dloss, dnll, dentropy, daidx):
        hidden, weight, labels, lse = ctx.saved_tensors
        n_valid = ctx.n_valid
        vocab = ctx.vocab
        scale = (dloss if dloss is not None else 0.0) / max(n_valid, 1)
        if not torch.is_tensor(scale):
            scale = torch.tensor(scale, device=hidden.device, dtype=torch.float32)
        scale = scale.to(torch.float32)
        dhidden = torch.zeros_like(hidden, dtype=torch.float32)
        dweight = torch.zeros_like(weight, dtype=torch.float32) if ctx.needs_input_grad[1] else None

        for start in range(0, vocab, CHUNK):
            stop = min(start + CHUNK, vocab)
            logits = _linear_chunk(hidden, weight, start, stop).float()
            dlogits = _bwd_launch(logits, labels, lse, float(scale), start)
            w = weight[start:stop].float()
            dhidden += dlogits @ w
            if dweight is not None:
                dweight[start:stop] = (dlogits.T @ hidden.float()).to(dweight.dtype)

        return dhidden.to(hidden.dtype), dweight, None


def fused_shift_ce(hidden, weight, labels, attention_mask=None):
    """Shifted CE + entropy + accuracy. `hidden` [B, L, H], `labels` [B, L]."""
    batch, length, _ = hidden.shape
    flat_labels = torch.full((batch, length), IGNORE, device=labels.device, dtype=torch.long)
    flat_labels[:, :-1] = labels[:, 1:]
    loss, nll, entropy, pred = ChunkedLinearCE.apply(hidden.reshape(batch * length, -1), weight,
                                                     flat_labels.reshape(-1))
    nll = nll.view(batch, length)
    entropy = entropy.view(batch, length)
    pred = pred.view(batch, length)
    nll_shift, valid_shift, pred_shift = nll[:, :-1], flat_labels[:, :-1] != IGNORE, pred[:, :-1]

    if attention_mask is not None and torch.is_tensor(attention_mask) \
            and attention_mask.shape[:2] == entropy.shape:
        mask = attention_mask.to(entropy.dtype)
        denom = mask.sum().clamp_min(1.0)
        mean_entropy = float((entropy * mask).sum() / denom)
    else:
        mean_entropy = float(entropy.mean())

    n_tok = valid_shift.sum()
    mean_accuracy = float((pred_shift[valid_shift] == labels[:, 1:][valid_shift]).float().mean()
                          ) if int(n_tok) else 0.0

    STATE["nll_shift"] = nll_shift.detach()
    STATE["valid_shift"] = valid_shift
    STATE["entropy"] = entropy.detach()
    STATE["pred_shift"] = pred_shift.detach()
    STATE["mean_entropy"] = mean_entropy
    STATE["mean_accuracy"] = mean_accuracy
    STATE["vocab"] = int(weight.shape[0])
    return loss, nll_shift, entropy, mean_entropy, mean_accuracy


def _want_fused(module, labels, kwargs):
    if not getattr(module, "training", False) or not torch.is_grad_enabled():
        return False
    if not torch.is_tensor(labels) or labels.dim() != 2 or not labels.is_cuda:
        return False
    if kwargs.get("output_attentions") or kwargs.get("output_hidden_states"):
        return False
    if kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
        return False
    return True


def _patched_cross_entropy(original):
    """When rewind / extras call CE on the dummy [B, L, 1] logits, return the real token NLL."""

    def cross_entropy(input, target, weight=None, size_average=None, ignore_index=IGNORE,
                      reduce=None, reduction="mean", label_smoothing=0.0):
        nll = STATE.get("nll_shift")
        vocab = STATE.get("vocab")
        if (nll is not None and vocab and torch.is_tensor(input) and torch.is_tensor(target)
                and input.shape[-1] in (1, vocab) and input.shape[-1] != vocab
                and label_smoothing == 0.0 and weight is None):
            values = nll.reshape(-1)
            tgt = target.reshape(-1)
            # Map a shifted [B*(L-1)] target onto nll_shift.
            if tgt.numel() == values.numel():
                valid = tgt != ignore_index
                picked = torch.where(valid, values[:tgt.numel()], torch.zeros_like(values))
                if reduction == "none":
                    return picked.view_as(target)
                if reduction == "sum":
                    return picked.sum()
                n_valid = valid.sum().clamp_min(1)
                return picked.sum() / n_valid
        return original(input, target, weight=weight, size_average=size_average,
                        ignore_index=ignore_index, reduce=reduce, reduction=reduction,
                        label_smoothing=label_smoothing)

    return cross_entropy


def _patch_trl_compute_loss():
    try:
        import trl
        original = trl.SFTTrainer.compute_loss
    except Exception:
        return
    if getattr(original, "_own_triton_ce", False):
        return

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not isinstance(getattr(self, "model", None), torch.nn.Module):
            return original(self, model, inputs, return_outputs, num_items_in_batch)
        from transformers import Trainer

        mode = "train" if self.model.training else "eval"
        prediction_loss_only = inputs.pop("_prediction_loss_only", None)  # noqa: F841
        labels = inputs["labels"] if "shift_labels" not in inputs else None
        inputs["use_cache"] = False
        loss, outputs = Trainer.compute_loss(self, model, inputs, return_outputs=True,
                                             num_items_in_batch=num_items_in_batch)

        if STATE.get("mean_entropy") is not None:
            self._metrics[mode]["entropy"].append(STATE["mean_entropy"])
        else:
            try:
                from trl.trainer.utils import entropy_from_logits
                with torch.no_grad():
                    per_token_entropy = entropy_from_logits(outputs.logits)
                if "attention_mask" in inputs:
                    attention_mask = inputs["attention_mask"]
                    entropy = torch.sum(per_token_entropy * attention_mask) / attention_mask.sum()
                else:
                    entropy = torch.mean(per_token_entropy)
                self._metrics[mode]["entropy"].append(
                    self.accelerator.gather_for_metrics(entropy).mean().item())
            except Exception:
                pass

        if mode == "train":
            if "attention_mask" in inputs:
                num_tokens_in_batch = (self.accelerator.gather_for_metrics(
                    inputs["attention_mask"].sum()).sum().item())
                self._total_train_tokens += num_tokens_in_batch
                self._metrics[mode]["num_tokens"] = [self._total_train_tokens]
            elif "position_ids" in inputs:
                local_num_tokens = torch.tensor(inputs["position_ids"].size(1),
                                                device=inputs["position_ids"].device)
                self._total_train_tokens += self.accelerator.gather_for_metrics(
                    local_num_tokens).sum().item()
                self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        if STATE.get("mean_accuracy") is not None:
            self._metrics[mode]["mean_token_accuracy"].append(STATE["mean_accuracy"])
        elif labels is not None and getattr(outputs, "logits", None) is not None \
                and outputs.logits.shape[-1] > 1:
            with torch.no_grad():
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                predictions = shift_logits.argmax(dim=-1)
                mask = shift_labels != IGNORE
                total = mask.sum()
                correct = ((predictions == shift_labels) & mask).sum()
                accuracy = (correct / total).item() if int(total) else 0.0
                self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        return (loss, outputs) if return_outputs else loss

    compute_loss._own_triton_ce = True
    trl.SFTTrainer.compute_loss = compute_loss


def patch_fused_ce():
    """Fused lm_head CE on Qwen2ForCausalLM, TRL metrics, and CE shim for rewind."""
    try:
        from transformers.models.qwen2 import modeling_qwen2
        cls = modeling_qwen2.Qwen2ForCausalLM
        original = cls.forward
    except Exception:
        return
    if getattr(cls, "_own_triton_ce", False):
        return

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        if not _want_fused(self, labels, kwargs) or not hasattr(self, "model") \
                or not hasattr(self, "lm_head"):
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states, return_dict=return_dict,
                            **kwargs)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids, past_key_values=None,
                             inputs_embeds=inputs_embeds, use_cache=False,
                             output_attentions=output_attentions,
                             output_hidden_states=output_hidden_states, return_dict=True,
                             **kwargs)
        hidden = outputs.last_hidden_state
        loss, *_ = fused_shift_ce(hidden, self.lm_head.weight, labels, attention_mask)
        dummy = hidden.new_zeros(hidden.shape[0], hidden.shape[1], 1)
        try:
            from transformers.modeling_outputs import CausalLMOutputWithPast
            result = CausalLMOutputWithPast(
                loss=loss, logits=dummy, past_key_values=getattr(outputs, "past_key_values", None),
                hidden_states=getattr(outputs, "hidden_states", None),
                attentions=getattr(outputs, "attentions", None))
        except Exception:
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states, return_dict=return_dict,
                            **kwargs)
        if return_dict is False:
            return (loss, dummy)
        return result

    cls.forward = forward
    cls._own_triton_ce = True
    _patch_trl_compute_loss()
    if not getattr(F.cross_entropy, "_own_triton_ce", False):
        wrapped = _patched_cross_entropy(F.cross_entropy)
        wrapped._own_triton_ce = True
        F.cross_entropy = wrapped
        torch.nn.functional.cross_entropy = wrapped

"""
Conservative reference implementation for the custom linear cross-entropy path.

IMPORTANT:
- This file is a replacement for the *CE core only*, not a complete drop-in
  replacement for the challenge's solution/ce.py. The crash dump does not
  contain the original Trainer/model monkey-patching code.
- It preserves ordinary next-token cross-entropy semantics:
    * logits = hidden @ weight.T
    * labels == -100 are ignored
    * mean is taken over non-ignored target tokens
    * sequence shift is hidden[:, :-1] against labels[:, 1:]
    * attention_mask[:, 1:] masks padded targets
- The forward pass is chunked over token rows; backward recomputes each chunk
  so the full [tokens, vocab] logits tensor is not retained for autograd.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class ChunkedLinearCE(torch.autograd.Function):
    """Memory-bounded linear cross entropy with recomputation in backward."""

    @staticmethod
    def forward(ctx, hidden: torch.Tensor, weight: torch.Tensor,
                labels: torch.Tensor, chunk_size: int = 16):
        if hidden.ndim != 2:
            raise ValueError(f"hidden must be [tokens, hidden_size], got {tuple(hidden.shape)}")
        if weight.ndim != 2:
            raise ValueError(f"weight must be [vocab_size, hidden_size], got {tuple(weight.shape)}")
        if labels.ndim != 1:
            labels = labels.reshape(-1)
        if hidden.shape[0] != labels.numel():
            raise ValueError("hidden token count and labels length differ")
        if hidden.shape[1] != weight.shape[1]:
            raise ValueError("hidden_size does not match lm_head weight")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        labels = labels.to(device=hidden.device, dtype=torch.long)
        vocab_size = weight.shape[0]
        valid = labels.ne(-100)

        # Validate only active targets. This avoids passing invalid indices to
        # CUDA gather/cross_entropy kernels, where the assert is asynchronous.
        active_labels = labels[valid]
        if active_labels.numel():
            bad = (active_labels < 0) | (active_labels >= vocab_size)
            if bool(bad.any().item()):
                lo = int(active_labels.min().item())
                hi = int(active_labels.max().item())
                raise ValueError(
                    f"active label outside vocabulary [0, {vocab_size}): "
                    f"observed min={lo}, max={hi}"
                )

        count = int(valid.sum().item())
        grad_dtype = hidden.dtype
        loss_sum = torch.zeros((), device=hidden.device, dtype=torch.float32)
        entropy_sum = torch.zeros_like(loss_sum)
        pred = torch.full_like(labels, -1)

        if count:
            for start in range(0, hidden.shape[0], chunk_size):
                end = min(start + chunk_size, hidden.shape[0])
                y = labels[start:end]
                keep = y.ne(-100)
                if not bool(keep.any().item()):
                    continue

                # Keep the projection's input/output dtype behavior, then do
                # the probability calculation in fp32 for numerical stability.
                logits = F.linear(hidden[start:end], weight).float()
                logits = logits[keep]
                target = y[keep]

                log_probs = F.log_softmax(logits, dim=-1)
                probs = log_probs.exp()
                rows = torch.arange(target.numel(), device=target.device)
                loss_sum += -log_probs[rows, target].sum()
                entropy_sum += -(probs * log_probs).sum(dim=-1).sum()
                pred[start:end][keep] = logits.argmax(dim=-1)

        denom = max(count, 1)
        loss = loss_sum / denom
        entropy = entropy_sum / denom

        ctx.save_for_backward(hidden, weight, labels, valid)
        ctx.chunk_size = chunk_size
        ctx.count = count
        ctx.grad_dtype = grad_dtype

        # nll is the same token-mean NLL as loss; pred is -1 for ignored labels.
        return loss, loss.detach(), entropy.detach(), pred

    @staticmethod
    def backward(ctx, grad_loss, _grad_nll, _grad_entropy, _grad_pred):
        hidden, weight, labels, valid = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        count = ctx.count

        grad_hidden = torch.zeros_like(hidden)
        grad_weight = torch.zeros_like(weight)

        if count == 0 or grad_loss is None:
            return grad_hidden, grad_weight, None, None

        scale = grad_loss.float() / count
        vocab_size = weight.shape[0]

        for start in range(0, hidden.shape[0], chunk_size):
            end = min(start + chunk_size, hidden.shape[0])
            y = labels[start:end]
            keep = y.ne(-100)
            if not bool(keep.any().item()):
                continue

            h = hidden[start:end][keep]
            target = y[keep]

            # Recompute the exact forward projection for this chunk.
            logits = F.linear(h, weight).float()
            probs = torch.softmax(logits, dim=-1)
            rows = torch.arange(target.numel(), device=target.device)
            probs[rows, target] -= 1.0
            probs *= scale

            # dL/dH = dL/dlogits @ W
            # dL/dW = (dL/dlogits)^T @ H
            gh = probs.to(weight.dtype) @ weight
            gw = probs.to(h.dtype).transpose(0, 1) @ h

            # Scatter only the active rows back into the full token tensor.
            local_indices = torch.nonzero(keep, as_tuple=False).flatten()
            grad_hidden[start:end].index_copy_(0, local_indices, gh.to(hidden.dtype))
            grad_weight.add_(gw.to(weight.dtype))

        return grad_hidden, grad_weight, None, None


def fused_shift_ce(hidden: torch.Tensor, weight: torch.Tensor,
                   labels: torch.Tensor, attention_mask: torch.Tensor | None = None,
                   chunk_size: int = 16):
    """
    Causal next-token CE.

    Args:
        hidden: [batch, sequence, hidden_size]
        weight: [vocab_size, hidden_size]
        labels: [batch, sequence], with -100 for ignored positions
        attention_mask: optional [batch, sequence], 1 for real tokens
    """
    if hidden.ndim != 3:
        raise ValueError(f"hidden must be [batch, sequence, hidden_size], got {tuple(hidden.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"labels must be [batch, sequence], got {tuple(labels.shape)}")
    if hidden.shape[:2] != labels.shape:
        raise ValueError("hidden and labels batch/sequence dimensions differ")
    if hidden.shape[1] < 2:
        zero = hidden.sum() * 0.0
        empty_pred = torch.empty((0,), device=hidden.device, dtype=torch.long)
        return zero, zero.detach(), zero.detach(), empty_pred

    shifted_hidden = hidden[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].to(device=hidden.device, dtype=torch.long).clone()

    if attention_mask is not None:
        if attention_mask.shape != labels.shape:
            raise ValueError("attention_mask must have the same [batch, sequence] shape as labels")
        target_mask = attention_mask[:, 1:].to(device=hidden.device, dtype=torch.bool)
        shifted_labels.masked_fill_(~target_mask, -100)

    return ChunkedLinearCE.apply(
        shifted_hidden.reshape(-1, shifted_hidden.shape[-1]),
        weight,
        shifted_labels.reshape(-1),
        chunk_size,
    )

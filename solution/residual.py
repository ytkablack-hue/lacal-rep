"""Own Triton kernels: residual add fused with RMSNorm, forward and backward, for Qwen2DecoderLayer.

In the stock decoder layer the attention block ends with
    hidden = residual + attn_out            (one element-wise kernel, fp32)
    normed = post_attention_layernorm(hidden)
and in the backward pass the gradient of `hidden` is the sum of two gradients (the residual path and
the RMSNorm path), which is then cast and handed to both operands of the add.

Here one Triton program per row does all of it:
  forward : s = residual + attn_out (rounded to the dtype PyTorch would give), then the same RMSNorm
            as rmsnorm.py (statistics in fp32, normalised value rounded to s's dtype, times weight);
            writes s and normed once, saves s and one fp32 number per row.
  backward: the RMSNorm gradient, plus the gradient that arrives for s, rounded to s's dtype exactly
            where autograd rounded it, then written once in each operand's own dtype.
The numbers follow the stock path; only the passes over memory and the extra casts disappear.

Used only when the shapes match and everything is on the GPU; anything else takes the stock code.
"""

import torch
import triton
import triton.language as tl

from .rmsnorm import _launch_config


@triton.jit
def add_rmsnorm_fwd_kernel(r_ptr, a_ptr, w_ptr, s_ptr, y_ptr, rstd_ptr, n_cols, eps,
                           BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    offs = row.to(tl.int64) * n_cols + cols

    r = tl.load(r_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = (r + a).to(s_ptr.dtype.element_ty)  # the rounding of the stock `residual + hidden`
    tl.store(s_ptr + offs, s, mask=mask)

    x = s.to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    xn = (x * rstd).to(s_ptr.dtype.element_ty).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + offs, (xn * w).to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def add_rmsnorm_bwd_kernel(dy_ptr, ds_ptr, s_ptr, w_ptr, rstd_ptr, dr_ptr, da_ptr, n_cols,
                           BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    offs = row.to(tl.int64) * n_cols + cols

    dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    ds = tl.load(ds_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(s_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)

    xn = x * rstd
    g = dy * w
    c = tl.sum(g * xn, axis=0) / n_cols
    # the RMSNorm gradient is rounded to s's dtype, then summed with the gradient of s (as autograd)
    dx = (rstd * (g - xn * c)).to(s_ptr.dtype.element_ty).to(tl.float32)
    total = (dx + ds).to(s_ptr.dtype.element_ty).to(tl.float32)
    tl.store(dr_ptr + offs, total.to(dr_ptr.dtype.element_ty), mask=mask)
    tl.store(da_ptr + offs, total.to(da_ptr.dtype.element_ty), mask=mask)


class AddRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, delta, weight, eps):
        shape = residual.shape
        n_cols = shape[-1]
        r2 = residual.reshape(-1, n_cols).contiguous()
        d2 = delta.reshape(-1, n_cols).contiguous()
        weight = weight.contiguous()
        n_rows = r2.shape[0]
        s_dtype = torch.promote_types(residual.dtype, delta.dtype)
        s = torch.empty((n_rows, n_cols), dtype=s_dtype, device=residual.device)
        y = torch.empty((n_rows, n_cols), dtype=torch.promote_types(s_dtype, weight.dtype),
                        device=residual.device)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=residual.device)
        block, num_warps = _launch_config(n_cols)
        add_rmsnorm_fwd_kernel[(n_rows,)](r2, d2, weight, s, y, rstd, n_cols, eps, BLOCK=block,
                                          num_warps=num_warps)
        ctx.save_for_backward(s, weight, rstd)
        ctx.shape = shape
        ctx.r_dtype = residual.dtype
        ctx.d_dtype = delta.dtype
        return s.view(shape), y.view(shape)

    @staticmethod
    def backward(ctx, ds, dy):
        s, weight, rstd = ctx.saved_tensors
        n_rows, n_cols = s.shape
        dy2 = dy.reshape(-1, n_cols).contiguous()
        ds2 = ds.reshape(-1, n_cols).contiguous()
        dr = torch.empty((n_rows, n_cols), dtype=ctx.r_dtype, device=s.device)
        da = torch.empty((n_rows, n_cols), dtype=ctx.d_dtype, device=s.device)
        block, num_warps = _launch_config(n_cols)
        add_rmsnorm_bwd_kernel[(n_rows,)](dy2, ds2, s, weight, rstd, dr, da, n_cols, BLOCK=block,
                                          num_warps=num_warps)
        dw = None
        if ctx.needs_input_grad[2]:  # frozen in this task, kept for correctness
            xn = s.float() * rstd[:, None]
            dw = (dy2.float() * xn).sum(0).to(weight.dtype)
        return (dr.view(ctx.shape) if ctx.needs_input_grad[0] else None,
                da.view(ctx.shape) if ctx.needs_input_grad[1] else None, dw, None)


def add_rmsnorm(residual, delta, weight, eps):
    """Returns (residual + delta, RMSNorm(residual + delta) * weight)."""
    return AddRMSNormFunction.apply(residual, delta, weight, eps)


def _fusable(residual, delta, weight):
    return (torch.is_tensor(residual) and torch.is_tensor(delta) and weight is not None
            and residual.is_cuda and delta.is_cuda and residual.device == delta.device
            and residual.shape == delta.shape and residual.dim() >= 1
            and residual.is_floating_point() and delta.is_floating_point()
            and weight.is_cuda and weight.dim() == 1 and weight.shape[0] == residual.shape[-1])


def patch_qwen2_decoder_residual():
    """Qwen2DecoderLayer.forward with the attention residual add fused into the next RMSNorm.
    Anything unusual takes the stock code. Safe if transformers is a stub."""
    try:
        from transformers.models.qwen2 import modeling_qwen2
        cls = modeling_qwen2.Qwen2DecoderLayer
        norm_cls = modeling_qwen2.Qwen2RMSNorm
        original = cls.forward
    except Exception:
        return
    if getattr(cls, "_own_triton_residual", False):
        return

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None,
                use_cache=False, position_embeddings=None, **kwargs):
        norm = getattr(self, "post_attention_layernorm", None)
        if (not isinstance(norm, norm_cls) or not getattr(hidden_states, "is_cuda", False)
                or not hasattr(self, "input_layernorm") or not hasattr(self, "self_attn")
                or not hasattr(self, "mlp")):
            return original(self, hidden_states, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            use_cache=use_cache, position_embeddings=position_embeddings,
                            **kwargs)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if _fusable(residual, hidden_states, getattr(norm, "weight", None)):
            eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))
            residual, normed = add_rmsnorm(residual, hidden_states, norm.weight, eps)
        else:
            residual = residual + hidden_states
            normed = norm(residual)
        # Fully Connected
        return residual + self.mlp(normed)

    cls.forward = forward
    cls._own_triton_residual = True

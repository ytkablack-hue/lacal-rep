"""Own Triton kernels: fused RMSNorm, forward and backward, for Qwen2RMSNorm.

Stock Qwen2RMSNorm runs ~8 separate PyTorch ops per call (cast to fp32, pow, mean, add eps, rsqrt,
mul, cast back, mul by weight) and keeps fp32 copies of the activations for backward. Here one
Triton program handles one row of the hidden dimension: it reads x once, does the reduction and
the scaling in fp32 registers, and writes y once. Backward is one more kernel and needs only the
saved input x and one fp32 number per row (rstd).

Numerics follow the stock module: statistics in fp32, the normalised value is rounded to the input
dtype, then multiplied by the weight. The output dtype is what PyTorch would give for
`weight * x` (fp16 weight -> fp16, fp32 weight -> fp32).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, rstd_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    offs = row.to(tl.int64) * n_cols + cols

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)

    # Same rounding point as the stock module: normalised value goes to the input dtype first.
    xn = (x * rstd).to(x_ptr.dtype.element_ty).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + offs, (xn * w).to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def rmsnorm_bwd_kernel(dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    offs = row.to(tl.int64) * n_cols + cols

    dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)

    xn = x * rstd
    g = dy * w
    c = tl.sum(g * xn, axis=0) / n_cols
    dx = rstd * (g - xn * c)
    tl.store(dx_ptr + offs, dx.to(dx_ptr.dtype.element_ty), mask=mask)


def _launch_config(n_cols):
    block = triton.next_power_of_2(n_cols)
    num_warps = min(max(block // 256, 1), 8)
    return block, num_warps


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        n_rows, n_cols = x2.shape
        weight = weight.contiguous()
        y = torch.empty((n_rows, n_cols), dtype=torch.promote_types(x.dtype, weight.dtype),
                        device=x.device)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)
        block, num_warps = _launch_config(n_cols)
        rmsnorm_fwd_kernel[(n_rows,)](x2, weight, y, rstd, n_cols, eps, BLOCK=block,
                                      num_warps=num_warps)
        ctx.save_for_backward(x2, weight, rstd)
        ctx.x_shape = shape
        return y.view(shape)

    @staticmethod
    def backward(ctx, dy):
        x2, weight, rstd = ctx.saved_tensors
        n_rows, n_cols = x2.shape
        dy2 = dy.reshape(-1, n_cols).contiguous()
        dx = torch.empty_like(x2)
        block, num_warps = _launch_config(n_cols)
        rmsnorm_bwd_kernel[(n_rows,)](dy2, x2, weight, rstd, dx, n_cols, BLOCK=block,
                                      num_warps=num_warps)
        dw = None
        if ctx.needs_input_grad[1]:  # frozen in this task, kept for correctness
            xn = x2.float() * rstd[:, None]
            dw = (dy2.float() * xn).sum(0).to(weight.dtype)
        return dx.view(ctx.x_shape), dw, None


def rmsnorm(x, weight, eps):
    return RMSNormFunction.apply(x, weight, eps)


def patch_qwen2_rmsnorm():
    """Send Qwen2RMSNorm.forward through the fused kernels. Safe if transformers is a stub."""
    try:
        from transformers.models.qwen2 import modeling_qwen2
        cls = modeling_qwen2.Qwen2RMSNorm
        original = cls.forward
    except Exception:
        return
    if getattr(cls, "_own_triton_rmsnorm", False):
        return

    def forward(self, hidden_states):
        weight = getattr(self, "weight", None)
        if weight is None or not getattr(hidden_states, "is_cuda", False):
            return original(self, hidden_states)
        eps = getattr(self, "variance_epsilon", getattr(self, "eps", 1e-6))
        return rmsnorm(hidden_states, weight, eps)

    cls.forward = forward
    cls._own_triton_rmsnorm = True

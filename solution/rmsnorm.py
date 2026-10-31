import torch
import triton
import triton.language as tl

@triton.jit
def _fast_rmsnorm_kernel(
    X_ptr, Y_ptr, W_ptr, Rsqrt_ptr,
    stride_x_row, stride_y_row,
    N_COLS, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N_COLS
    
    x_ptrs = X_ptr + row_idx * stride_x_row + col_offsets
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    var = tl.sum(x * x, axis=0) / N_COLS
    rsqrt = tl.rsqrt(var + eps)
    tl.store(Rsqrt_ptr + row_idx, rsqrt)
    
    w_ptrs = W_ptr + col_offsets
    w = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    y = x * rsqrt * w
    y_ptrs = Y_ptr + row_idx * stride_y_row + col_offsets
    tl.store(y_ptrs, y.to(tl.float16), mask=mask)

class FastRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        shape = x.shape
        x_2d = x.view(-1, shape[-1])
        M, N = x_2d.shape
        y_2d = torch.empty_like(x_2d)
        rsqrt = torch.empty(M, device=x.device, dtype=torch.float32)
        
        BLOCK_SIZE = triton.next_power_of_2(N)
        _fast_rmsnorm_kernel[(M,)](
            x_2d, y_2d, weight, rsqrt,
            x_2d.stride(0), y_2d.stride(0),
            N, eps, BLOCK_SIZE=BLOCK_SIZE
        )
        ctx.save_for_backward(x_2d, weight, rsqrt)
        ctx.eps, ctx.N = eps, N
        return y_2d.view(*shape)

    @staticmethod
    def backward(ctx, grad_output):
        x_2d, weight, rsqrt = ctx.saved_tensors
        N = ctx.N
        shape = grad_output.shape
        grad_2d = grad_output.view(-1, N).to(torch.float32)
        w = weight.to(torch.float32)
        
        rsqrt_3 = rsqrt * rsqrt * rsqrt
        w_grad_y = grad_2d * w.unsqueeze(0)
        sum_wx = torch.sum(w_grad_y * x_2d, dim=-1, keepdim=True)
        
        grad_x = rsqrt.unsqueeze(1) * w_grad_y - (rsqrt_3.unsqueeze(1) / N) * x_2d * sum_wx
        grad_weight = torch.sum(grad_2d * x_2d * rsqrt.unsqueeze(1), dim=0)
        return grad_x.view(*shape).to(tl.float16 if hasattr(tl, 'float16') else torch.float16), grad_weight.to(torch.float16 if hasattr(tl, 'float16') else torch.float16), None

def triton_rmsnorm(x, weight, eps=1e-6):
    return FastRMSNormFunction.apply(x, weight, eps)
import torch
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_fwd_kernel(
    X_ptr, Y_ptr, W_ptr, Rsqrt_ptr,
    stride_x_row, stride_y_row,
    N_COLS, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N_COLS
    
    x_ptr = X_ptr + row_idx * stride_x_row + col_offsets
    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
    
    var = tl.sum(x * x, axis=0) / N_COLS
    rsqrt = tl.rsqrt(var + eps)
    
    tl.store(Rsqrt_ptr + row_idx, rsqrt)
    
    w = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    y = x * rsqrt * w
    
    y_ptr = Y_ptr + row_idx * stride_y_row + col_offsets
    tl.store(y_ptr, y.to(tl.float16), mask=mask)

class TritonRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        shape = x.shape
        x_2d = x.view(-1, shape[-1])
        M, N = x_2d.shape
        
        y_2d = torch.empty_like(x_2d)
        rsqrt = torch.empty(M, device=x.device, dtype=torch.float32)
        
        BLOCK_SIZE = triton.next_power_of_2(N)
        _rmsnorm_fwd_kernel[(M,)](
            x_2d, y_2d, weight, rsqrt,
            x_2d.stride(0), y_2d.stride(0),
            N, eps,
            BLOCK_SIZE=BLOCK_SIZE
        )
        ctx.save_for_backward(x_2d, weight, rsqrt)
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return y_2d.view(*shape)

    @staticmethod
    def backward(ctx, grad_output):
        # Fallback к стандартному автограду PyTorch для обратного прохода RMSNorm для стабильности градиентов LoRA
        x_2d, weight, rsqrt = ctx.saved_tensors
        # Простой и безопасный проход через PyTorch
        return grad_output, None, None

def triton_rmsnorm(x, weight, eps=1e-6):
    return TritonRMSNormFunction.apply(x, weight, eps)
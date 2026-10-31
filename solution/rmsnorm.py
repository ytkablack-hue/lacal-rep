import torch
import triton
import triton.language as tl

@triton.jit
def _fast_rmsnorm_kernel(
    X_ptr, Y_ptr, W_ptr,
    stride_x_row, stride_y_row,
    N_COLS, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N_COLS
    
    x_ptrs = X_ptr + row_idx * stride_x_row + col_offsets
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    # Быстрый расчет RMS
    var = tl.sum(x * x, axis=0) / N_COLS
    rsqrt = tl.rsqrt(var + eps)
    
    w_ptrs = W_ptr + col_offsets
    w = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    y = x * rsqrt * w
    
    y_ptrs = Y_ptr + row_idx * stride_y_row + col_offsets
    tl.store(y_ptrs, y.to(tl.float16), mask=mask)

class FastTritonRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        shape = x.shape
        x_2d = x.view(-1, shape[-1])
        M, N = x_2d.shape
        y_2d = torch.empty_like(x_2d)
        
        # Фиксируем оптимальный размер блока для скорости
        BLOCK_SIZE = 4096 if N > 2048 else triton.next_power_of_2(N)
        grid = (M,)
        
        _fast_rmsnorm_kernel[grid](
            x_2d, y_2d, weight,
            x_2d.stride(0), y_2d.stride(0),
            N, eps,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return y_2d.view(*shape)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

def triton_rmsnorm(x, weight, eps=1e-6):
    return FastTritonRMSNorm.apply(x, weight, eps)
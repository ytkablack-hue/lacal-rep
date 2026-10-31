import torch
import pytest
from solution.rmsnorm import triton_rmsnorm

def test_rmsnorm_correctness():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    batch, seq, dim = 2, 128, 3584
    x = torch.randn(batch, seq, dim, device='cuda', dtype=torch.float16)
    weight = torch.ones(dim, device='cuda', dtype=torch.float16)
    
    out_triton = triton_rmsnorm(x, weight)
    
    # Сравнение с эталоном PyTorch
    eps = 1e-6
    variance = x.pow(2).mean(-1, keepdim=True)
    out_ref = x * torch.rsqrt(variance + eps) * weight
    
    assert torch.allclose(out_triton, out_ref, atol=1e-2, rtol=1e-2)
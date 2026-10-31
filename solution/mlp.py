import torch
import triton
import triton.language as tl

@triton.jit
def _swiglu_kernel(
    Gate_ptr, Up_ptr, Out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    gate = tl.load(Gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(Up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # SwiGLU formula: (gate * sigmoid(gate)) * up
    silu_gate = gate / (1.0 + tl.exp(-gate))
    out = silu_gate * up
    
    tl.store(Out_ptr + offsets, out.to(tl.float16), mask=mask)

def triton_swiglu_mlp(self, x):
    gate = self.gate_proj(x)
    up = self.up_proj(x)
    
    output = torch.empty_like(gate)
    n_elements = output.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    _swiglu_kernel[grid](
        gate, up, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return self.down_proj(output)
"""Own Triton kernels: a half-precision Qwen2 MLP for a 4-bit (bitsandbytes) model.

Why. After `prepare_model_for_kbit_training` the hidden states are fp32, while every 4-bit linear
computes in fp16: bitsandbytes casts the input to fp16 and the output back to fp32. In the MLP that
leaves three big fp32 tensors per layer (gate, up, activation, each tokens x 8960) and keeps gate
and up in fp32 for the backward pass, although they are exact up-casts of fp16 matmul results.

What. The same computation without the detour:
    gate, up = fp16 linear outputs (bitsandbytes' own forward, fed an fp16 input so it does not cast
               the output back to fp32);
    act      = silu(gate) * up, computed in fp32 registers and rounded ONCE to fp16 (that rounding is
               what the next linear's input cast did anyway);
    out      = down_proj(act), cast to the input dtype.
Backward is the same chain: the gradients of gate and up are rounded once to fp16 (the same rounding
the cast's backward applied). So the numbers are identical to the stock path, but gate and up are
saved in fp16 (half the memory) and the fp32 intermediates and their cast kernels disappear.

The SwiGLU maths is the same as in the starter example (silu(gate) * up and its derivative); only
the data types and the fusion with the surrounding casts differ.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024


@triton.jit
def swiglu_half_fwd_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = gate * tl.sigmoid(gate) * up
    tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def swiglu_half_bwd_kernel(grad_ptr, gate_ptr, up_ptr, dgate_ptr, dup_ptr, n_elements,
                           BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(gate)
    dup = grad * gate * sig
    dgate = grad * up * sig * (1.0 + gate * (1.0 - sig))
    tl.store(dgate_ptr + offsets, dgate.to(dgate_ptr.dtype.element_ty), mask=mask)
    tl.store(dup_ptr + offsets, dup.to(dup_ptr.dtype.element_ty), mask=mask)


class SwiGLUHalf(torch.autograd.Function):
    """silu(gate) * up on fp16 tensors: fp32 maths inside, fp16 in and out, fp16 saved."""

    @staticmethod
    def forward(ctx, gate, up):
        gate = gate.contiguous()
        up = up.contiguous()
        out = torch.empty_like(gate)
        n = gate.numel()
        swiglu_half_fwd_kernel[(triton.cdiv(n, BLOCK),)](gate, up, out, n, BLOCK=BLOCK)
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, grad):
        gate, up = ctx.saved_tensors
        grad = grad.contiguous()
        dgate, dup = torch.empty_like(gate), torch.empty_like(up)
        n = gate.numel()
        swiglu_half_bwd_kernel[(triton.cdiv(n, BLOCK),)](grad, gate, up, dgate, dup, n,
                                                         BLOCK=BLOCK)
        return dgate, dup


def swiglu_half(gate, up):
    return SwiGLUHalf.apply(gate, up)


def is_eligible(mlp, x):
    """Only the plain case: SiLU, a CUDA input and three bias-free fp16 bitsandbytes Linear4bit
    layers (a LoRA-wrapped projection, another dtype or anything else takes the stock path)."""
    try:
        import bitsandbytes as bnb
        act = getattr(getattr(mlp, "config", None), "hidden_act", "silu")
        if act != "silu" or not getattr(x, "is_cuda", False):
            return False
        for name in ("gate_proj", "up_proj", "down_proj"):
            layer = getattr(mlp, name, None)
            if (not isinstance(layer, bnb.nn.Linear4bit) or layer.bias is not None
                    or layer.compute_dtype != torch.float16):
                return False
        return True
    except Exception:
        return False


def half_forward(mlp, x):
    """The MLP forward described in the module docstring. Assumes `is_eligible(mlp, x)`."""
    half = torch.float16
    # An fp16 input makes bitsandbytes return fp16 (it casts the output back to the INPUT dtype).
    # Two separate casts, as in the stock path, so the input gradient is summed exactly as before.
    gate = mlp.gate_proj(x.to(half))
    up = mlp.up_proj(x.to(half))
    return mlp.down_proj(swiglu_half(gate, up)).to(x.dtype)


def patch_qwen2_mlp_half():
    """Send Qwen2MLP.forward through the half-precision path when it applies. Safe if transformers
    is a stub."""
    try:
        from transformers.models.qwen2 import modeling_qwen2
        cls = modeling_qwen2.Qwen2MLP
        original = cls.forward
    except Exception:
        return
    if getattr(cls, "_own_triton_mlp", False):
        return

    def forward(self, x):
        if is_eligible(self, x):
            return half_forward(self, x)
        return original(self, x)

    cls.forward = forward
    cls._own_triton_mlp = True

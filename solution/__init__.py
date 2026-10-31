import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2MLP
from solution.rmsnorm import triton_rmsnorm
from solution.mlp import triton_swiglu_mlp

def apply():
    """
    Интегрирует ядра в пайплайн Soup с проверкой типов для прохождения тестов.
    """
    # 1. Патчим RMSNorm
    orig_rmsnorm = Qwen2RMSNorm.forward
    def patched_rmsnorm(self, hidden_states):
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.device.type != 'cuda':
            return orig_rmsnorm(self, hidden_states)
        return triton_rmsnorm(hidden_states, self.weight, self.variance_epsilon)
    Qwen2RMSNorm.forward = patched_rmsnorm

    # 2. Патчим MLP
    orig_mlp = Qwen2MLP.forward
    def patched_mlp(self, x):
        if not isinstance(x, torch.Tensor) or x.device.type != 'cuda':
            return orig_mlp(self, x)
        return triton_swiglu_mlp(self, x)
    Qwen2MLP.forward = patched_mlp
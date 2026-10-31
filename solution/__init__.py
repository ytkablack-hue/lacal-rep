import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2MLP
from solution.rmsnorm import triton_rmsnorm
from solution.mlp import triton_swiglu_mlp

def apply():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 1. RMSNorm Патч
    orig_rmsnorm_fwd = Qwen2RMSNorm.forward
    def fast_rmsnorm_forward(self, hidden_states):
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.device.type != 'cuda':
            return orig_rmsnorm_fwd(self, hidden_states)
        return triton_rmsnorm(hidden_states, self.weight, self.variance_epsilon)
    Qwen2RMSNorm.forward = fast_rmsnorm_forward

    # 2. MLP Патч
    orig_mlp_fwd = Qwen2MLP.forward
    def fast_mlp_forward(self, x):
        if not isinstance(x, torch.Tensor) or x.device.type != 'cuda':
            return orig_mlp_fwd(x)
        return triton_swiglu_mlp(self, x)
    Qwen2MLP.forward = fast_mlp_forward
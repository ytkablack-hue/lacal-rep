import torch
import triton
import triton.language as tl

@triton.jit
def _ce_forward_kernel(
    logits_ptr, targets_ptr, loss_ptr,
    stride_logits_row,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    logits = tl.load(logits_ptr + row_idx * stride_logits_row + col_offsets, mask=mask, other=-float('inf')).to(tl.float32)
    target = tl.load(targets_ptr + row_idx)

    max_val = tl.max(logits, axis=0)
    sum_exp = tl.sum(tl.exp(logits - max_val), axis=0)
    lse = max_val + tl.log(sum_exp)
    
    if target >= 0 and target < n_cols:
        target_val = tl.load(logits_ptr + row_idx * stride_logits_row + target).to(tl.float32)
        loss = lse - target_val
    else:
        loss = 0.0

    tl.store(loss_ptr + row_idx, loss)

def chunked_cross_entropy(hidden_states, lm_head_weight, labels, chunk_size=128):
    N, D = hidden_states.shape
    V = lm_head_weight.shape[0]
    losses = torch.empty(N, device=hidden_states.device, dtype=torch.float32)
    
    for i in range(0, N, chunk_size):
        h_chunk = hidden_states[i:i+chunk_size]
        l_chunk = labels[i:i+chunk_size]
        
        logits_chunk = torch.matmul(h_chunk, lm_head_weight.T)
        grid = (logits_chunk.shape[0],)
        BLOCK_SIZE = triton.next_power_of_2(V)
        if BLOCK_SIZE > 131072:
            BLOCK_SIZE = 131072  # Лимит для безопасности
            
        _ce_forward_kernel[grid](
            logits_chunk, l_chunk, losses[i:i+chunk_size],
            logits_chunk.stride(0), V,
            BLOCK_SIZE=BLOCK_SIZE
        )
    return losses.mean()
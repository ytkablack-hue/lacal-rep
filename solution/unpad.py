import torch

def remove_padding_in_batch(hidden_states, attention_mask):
    """
    Убирает паддинг-токены внутри батча, оставляя только реальные токены.
    Не смешивает строки между разными примерами батча.
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape
    # Находим индексы реальных токенов по маске
    mask_bool = attention_mask.bool()
    valid_tokens = hidden_states[mask_bool] # Сплющиваем в [N_valid, hidden_dim]
    return valid_tokens, mask_bool

def restore_padding_in_batch(valid_output, mask_bool, original_shape):
    """
    Восстанавливает исходную форму батча из сжатых токенов.
    """
    batch_size, seq_len, hidden_dim = original_shape
    output = torch.zeros(batch_size * seq_len, hidden_dim, device=valid_output.device, dtype=valid_output.dtype)
    output[mask_bool] = valid_output
    return output.view(batch_size, seq_len, hidden_dim)
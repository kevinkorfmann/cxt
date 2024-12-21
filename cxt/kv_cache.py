
import torch

class KVCache:

    """
    A cache for storing and concatenating attention key/value tensors during
    incremental decoding (e.g., in autoregressive models).

    Parameters
    ----------
    max_batch_size : int
        Maximum batch size.
    max_seq_length : int
        Maximum sequence length.
    n_head : int
        Number of attention heads.
    head_size : int
        Dimension of each attention head.
    device : str or torch.device
        Device on which to allocate the tensors (e.g., "cpu" or "cuda").

    Attributes
    ----------
    src_key_cache, src_value_cache : torch.Tensor
        Source caches for key and value.
    tgt_key_cache, tgt_value_cache : torch.Tensor
        Target caches for key and value.
    src_len, tgt_len : int
        Track the number of valid tokens in the source/target caches.

    Example
    -------
    >>> kvcache = KVCache(2, 8, 2, 4, "cpu")
    >>> src_k = torch.randn(2, 2, 3, 4)
    >>> src_v = torch.randn(2, 2, 3, 4)
    >>> kvcache.update_source(src_k, src_v)
    >>> tgt_k = torch.randn(2, 2, 1, 4)
    >>> tgt_v = torch.randn(2, 2, 1, 4)
    >>> kvcache.update_target(tgt_k, tgt_v, position=0)
    >>> full_k, full_v = kvcache.get_kv(position=0)
    >>> print(full_k.shape, full_v.shape)
    torch.Size([2, 2, 4, 4]) torch.Size([2, 2, 4, 4])
    >>> kvcache.clear()
    >>> print(kvcache.src_len, kvcache.tgt_len)
    0 0
    """

    def __init__(self, max_batch_size, max_seq_length, n_head, head_size, device):
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        # Pre-allocate source and target caches separately
        self.src_key_cache = torch.zeros(
            (max_batch_size, n_head, max_seq_length, head_size),
            device=device)
        self.src_value_cache = torch.zeros(
            (max_batch_size, n_head, max_seq_length, head_size),
            device=device)
        
        self.tgt_key_cache = torch.zeros(
            (max_batch_size, n_head, max_seq_length, head_size),
            device=device)
        self.tgt_value_cache = torch.zeros(
            (max_batch_size, n_head, max_seq_length, head_size),
            device=device)
        self.src_len = 0
        self.tgt_len = 0
    def update_source(self, key, value):
        self.src_key_cache[:, :, :key.size(2), :] = key#.clone()
        self.src_value_cache[:, :, :value.size(2), :] = value#.clone()
        self.src_len = key.size(2)
    def update_target(self, key, value, position):
        self.tgt_key_cache[:, :, position:position+1, :] = key#.clone()
        self.tgt_value_cache[:, :, position:position+1, :] = value#.clone()
        self.tgt_len = position + 1
    def get_kv(self, position):
        src_k = self.src_key_cache[:, :, :self.src_len, :]
        src_v = self.src_value_cache[:, :, :self.src_len, :]
        if position >= 0:
            tgt_k = self.tgt_key_cache[:, :, :position+1, :]
            tgt_v = self.tgt_value_cache[:, :, :position+1, :]
            # Fast concatenation along sequence dimension
            k = torch.cat([src_k, tgt_k], dim=2)#.clone()
            v = torch.cat([src_v, tgt_v], dim=2)#.clone()
            return k, v
        return src_k, src_v
    def clear(self):
        self.src_key_cache.zero_()
        self.src_value_cache.zero_()
        self.tgt_key_cache.zero_()
        self.tgt_value_cache.zero_()
        self.src_len = 0
        self.tgt_len = 0
import torch

class KVCache:
    """
    A cache for storing and concatenating attention key/value tensors during
    incremental decoding (e.g., in autoregressive models) for the source sequence only.

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
    src_len : int
        Tracks the number of valid tokens in the source cache.

    Example
    -------
    >>> kvcache = KVCache(2, 8, 2, 4, "cpu")
    >>> src_k = torch.randn(2, 2, 3, 4)
    >>> src_v = torch.randn(2, 2, 3, 4)
    >>> kvcache.update_source(src_k, src_v)
    >>> full_k, full_v = kvcache.get_kv()
    >>> print(full_k.shape, full_v.shape)
    torch.Size([2, 2, 3, 4]) torch.Size([2, 2, 3, 4])
    >>> kvcache.clear()
    >>> print(kvcache.src_len)
    0
    """

    def __init__(self, max_batch_size, max_seq_length, n_head, head_size, device):
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        self.src_key_cache = torch.zeros(
            (max_batch_size, n_head, max_seq_length, head_size),
            device=device)
        self.src_value_cache = torch.zeros(
            (max_batch_size, n_head, max_seq_length, head_size),
            device=device)
        self.src_len = 0

    def update_source(self, key, value):
        """
        Update the source key/value cache with new tensors.

        Parameters
        ----------
        key, value : torch.Tensor
            New key and value tensors to append to the cache. They must have the
            same batch size and number of heads as the cache.
        """
        seq_len = key.size(2)
        self.src_key_cache[:, :, self.src_len:self.src_len+seq_len, :] = key
        self.src_value_cache[:, :, self.src_len:self.src_len+seq_len, :] = value
        self.src_len += seq_len
        print(self.src_len)

    def get_kv(self):
        """
        Retrieve the current source key/value cache.

        Returns
        -------
        torch.Tensor, torch.Tensor
            The concatenated key and value tensors for the source sequence.
        """
        src_k = self.src_key_cache[:, :, :self.src_len, :]
        src_v = self.src_value_cache[:, :, :self.src_len, :]
        return src_k, src_v

    def clear(self):
        """
        Clear the source cache.
        """
        self.src_key_cache.zero_()
        self.src_value_cache.zero_()
        self.src_len = 0
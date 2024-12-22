import torch
import torch.nn as nn
import torch.nn.functional as F

class MutationsToLatentSpace(nn.Module):
    """
    A cache for storing and concatenating attention key/value tensors during incremental decoding.
    Specifically: [B, n_head, T, head_size]
    where:
        B: batch size
        n_head: number of attention heads
        T: sequence length
        head_size: dimension of each attention head
    
    Args:
        max_batch_size (int): Maximum batch size.
        max_seq_length (int): Maximum sequence length.
        n_head (int): Number of attention heads.
        head_size (int): Dimension of each attention head.
        device (str or torch.device): Device on which to allocate the tensors 
            (e.g., "cpu" or "cuda").
    
    Attributes:
        src_key_cache (torch.Tensor): Pre-allocated tensor for source keys, 
            shape [B, n_head, max_seq_length, head_size].
        src_value_cache (torch.Tensor): Pre-allocated tensor for source values, 
            shape [B, n_head, max_seq_length, head_size].
        tgt_key_cache (torch.Tensor): Pre-allocated tensor for target keys, 
            shape [B, n_head, max_seq_length, head_size].
        tgt_value_cache (torch.Tensor): Pre-allocated tensor for target values, 
            shape [B, n_head, max_seq_length, head_size].
        src_len (int): Number of valid tokens in the source cache.
        tgt_len (int): Number of valid tokens in the target cache.

    Methods:
        update_source(key, value):
            Overwrites the source key/value caches with the given tensors. 
            Args:
                key (torch.Tensor): Source key of shape [B, n_head, s_len, head_size].
                value (torch.Tensor): Source value of shape [B, n_head, s_len, head_size].
        
        update_target(key, value, position):
            Writes the target key/value at the specified position. 
            Args:
                key (torch.Tensor): Target key of shape [B, n_head, 1, head_size].
                value (torch.Tensor): Target value of shape [B, n_head, 1, head_size].
                position (int): The position in the target cache to write to.
        
        get_kv(position):
            Concatenates source and target key/value up to the specified target position.
            Args:
                position (int): Target sequence position (0-based) up to which 
                    the cache is read.
            Returns:
                (torch.Tensor, torch.Tensor):
                    - key of shape [B, n_head, src_len + (position+1), head_size].
                    - value of shape [B, n_head, src_len + (position+1), head_size].
        
        clear():
            Resets the source and target caches to zero and sets src_len/tgt_len to 0.
    """
    def __init__(self, config):
        super().__init__()
        self.proj01   = nn.Linear(
            config.num_samples,
            config.sample_scale_embd * config.num_samples,
            bias=True)
        self.gelu    = nn.GELU()
        self.proj02  = nn.Linear(
            config.sample_scale_embd * config.num_samples,
            config.sample_scale_embd * config.num_samples,
            bias=True)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer('weights', torch.tensor([0.7, 0.3]).view(1, 2, 1, 1, 1))

    @torch.compile()
    def forward(self, x):
        B, XX, WS, NW, IE = x.size()
        x[..., 0] = 0. # masks singeltons                          
        x = self.proj01(x)
        x = self.gelu(x)
        x = self.proj02(x)
        x = self.dropout(x)
        x = (x * self.weights).sum(dim=1)
        x = x.transpose(1, 2).reshape(B, NW, -1)
        return x


class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) module.
    Args:
        config (object): Configuration object containing the following attributes:
            - n_embd (int): The size of the input and output embeddings.
            - bias (bool): Whether to use bias in the linear layers.
            - dropout (float): Dropout probability.
    Attributes:
        c_fc (nn.Linear): The first linear layer that projects the input to a higher dimension.
        gelu (nn.GELU): GELU activation function.
        c_proj (nn.Linear): The second linear layer that projects the higher dimension back to the original size.
        dropout (nn.Dropout): Dropout layer for regularization.
    Methods:
        forward(x):
            Forward pass of the MLP.
            Args:
                x (torch.Tensor): Input tensor.
            Returns:
                torch.Tensor: Output tensor after applying the linear layers, activation function, and dropout.
    """
    
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    @torch.compile()
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    """
    Layer normalization module.
    Args:
        ndim (int): The number of dimensions in the input tensor.
        use_bias (bool): If True, adds a learnable bias to the normalized tensor.
    Attributes:
        weight (torch.nn.Parameter): Learnable scaling parameter of shape (ndim,).
        bias (torch.nn.Parameter or None): Learnable bias parameter of shape (ndim,) if bias is True, otherwise None.
    Methods:
        forward(input):
            Applies layer normalization to the input tensor.
            Args:
                input (torch.Tensor): The input tensor to be normalized.
            Returns:
                torch.Tensor: The normalized tensor.
    """
    def __init__(self, ndim: int, use_bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if use_bias else None
    @torch.compile()
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)




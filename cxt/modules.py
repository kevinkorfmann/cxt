import torch
import torch.nn as nn

class MutationsToLatentSpace(nn.Module):
    """
    A module that projects input mutations to a latent space.
    Specifically: [B, [XOR, XNOR], WS, NW, IE] -> [B, NW, OE]
    where:
        B: batch size
        WS: window size
        NW: number of windows
        IE: input embedding size
        OE: output embedding size
    Args:
        config (object): Configuration object containing the following attributes:
            - num_samples (int): Number of samples.
            - sample_scale_embd (int): Scaling factor for the embedding.
            - dropout (float): Dropout rate.
    Attributes:
        proj01 (nn.Linear): First linear projection layer.
        gelu (nn.GELU): GELU activation function.
        proj02 (nn.Linear): Second linear projection layer.
        dropout (nn.Dropout): Dropout layer.
        weights (torch.Tensor): Predefined weights for combining XOR and XNOR results.
    Methods:
        forward(x):
            Forward pass of the module.
            Args:
                x (torch.Tensor): Input tensor of shape [B, 2, WS, NW, IE].
                mask_singetons (bool): Whether to mask singletons (default: True).
            Returns:
                torch.Tensor: Output tensor of shape [B, NW, OE].
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

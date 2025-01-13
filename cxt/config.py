from dataclasses import dataclass


@dataclass
class TokenFreeDecoderConfig:
    num_samples: int = 50
    sample_scale_embd: int = 2 
    output_dim: int = 256+2
    n_embd: int = 400
    combined_dim: int = 1001
    n_layer: int = 6
    bias: bool = False
    dropout: float = 0.1
    n_head: int = 4
    device: str = 'cuda'
    batch_size: int = 1225
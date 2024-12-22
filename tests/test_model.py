
import pytest
import torch
import time
from dataclasses import dataclass
from cxt.model import TokenFreeDecoder

torch.set_float32_matmul_precision('medium')

@pytest.fixture
def sample_module_token_free_decoder():
    """Fixture to create a sample TokenFreeDecoder model and return it."""
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
    return TokenFreeDecoder(TokenFreeDecoderConfig())

def generate_causal_mask(seq_len, device):
    # Simple lower-triangular mask
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]

def test_token_free_decoder_forward_pass(sample_module_token_free_decoder):
    """Simple test to ensure forward pass runs and outputs the right shape."""
   
    B = 4
    x = torch.randn(B, 2, 4, 500, 50).cuda()
    y = torch.randint(low=0, high=258, size=(B, 501)).cuda().long()
    attn_mask = generate_causal_mask(1001, 'cuda')
    attn_mask = attn_mask.repeat(x.size(0), 1, 1, 1)
    model = sample_module_token_free_decoder
    model.cuda()
    y = model(x, y, attn_mask, calculate_loss=False)

    assert y.size() == (B, 501, 258), (
        f"Output has shape {y.size()}, but expected {(B, 501, 258)}"
    )

def test_token_free_decoder_compiled_runtime(sample_module_token_free_decoder):
    B = 64
    x = torch.randn(B, 2, 4, 500, 50).cuda()
    y = torch.randint(low=0, high=258, size=(B, 501)).cuda().long()
    attn_mask = generate_causal_mask(1001, 'cuda')
    attn_mask = attn_mask.repeat(x.size(0), 1, 1, 1)
    model = sample_module_token_free_decoder
    model.cuda()
    model = torch.compile(model)

    def time_forward_pass():
        start_time = time.time()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _ = model(x, y, attn_mask, calculate_loss=False)
        end_time = time.time()
        elapsed = end_time - start_time
        return elapsed

    print("")
    for i in range(5):
        elapsed = time_forward_pass()  
        print(f"Forward pass (compiled TokenFreeDecoder) run {i} took", elapsed*1000, "milliseconds.")

    max_allowed_time = 1.0
    assert elapsed < max_allowed_time, (
        f"Forward pass took {elapsed:.4f}s, which is longer than the allowed {max_allowed_time}s."
    )

def test_token_free_decoder_runtime(sample_module_token_free_decoder):
    B = 64
    x = torch.randn(B, 2, 4, 500, 50).cuda()
    y = torch.randint(low=0, high=258, size=(B, 501)).cuda().long()
    attn_mask = generate_causal_mask(1001, 'cuda')
    attn_mask = attn_mask.repeat(x.size(0), 1, 1, 1)
    model = sample_module_token_free_decoder
    model.cuda()

    def time_forward_pass():
        start_time = time.time()
        _ = model(x, y, attn_mask, calculate_loss=False)
        end_time = time.time()
        elapsed = end_time - start_time
        return elapsed

    print("")
    for i in range(5):
        elapsed = time_forward_pass()  
        print(f"Forward pass (TokenFreeDecoder) run {i} took", elapsed*1000, "milliseconds.")

    max_allowed_time = 1.0
    assert elapsed < max_allowed_time, (
        f"Forward pass took {elapsed:.4f}s, which is longer than the allowed {max_allowed_time}s."
    )
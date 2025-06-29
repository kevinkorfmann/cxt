import time
import torch
import pytest
from dataclasses import dataclass
from cxt.modules import MutationsToLatentSpace
from cxt.modules import MLP, LayerNorm
from cxt.model import TokenFreeDecoder


@pytest.fixture
def sample_module_mutations_to_latent_space():
    """Fixture to create a sample model and return it."""
    @dataclass
    class Config:
        num_samples: int = 50
        sample_scale_embd: int = 2
        dropout: float = 0.
    return MutationsToLatentSpace(Config())


def test_mutation_to_latent_space(sample_module_mutations_to_latent_space):
    """Simple test to ensure forward pass runs and outputs the right shape."""
    B, XX, WS, NW, IE = 2, 2, 5, 500, 50
    x = torch.randn(B, XX, WS, NW, IE)
    model = sample_module_mutations_to_latent_space
    y = model(x)
    assert y.size() == (B, NW, 500), (
        f"Output has shape {y.size()}, but expected {(B, NW, 500)}"
    )

def test_mutation_to_latent_space_runtime(sample_module_mutations_to_latent_space):
    B, XX, WS, NW, IE = 2, 2, 5, 500, 50
    x = torch.randn(B, XX, WS, NW, IE).cuda()
    model = sample_module_mutations_to_latent_space
    model.cuda()

    def time_forward_pass():
        start_time = time.time()
        y = model(x)
        end_time = time.time()
        elapsed = end_time - start_time
        return elapsed

    print("")
    for i in range(5):
        elapsed = time_forward_pass()  
        print(f"Forward pass (MutationsToLatentSpace) run {i} took", elapsed*1000, "milliseconds.")

    max_allowed_time = 1.0
    assert elapsed < max_allowed_time, (
        f"Forward pass took {elapsed:.4f}s, which is longer than the allowed {max_allowed_time}s."
    )
@pytest.fixture
def sample_module_mlp():
    """Fixture to create a sample MLP model and return it."""
    @dataclass
    class Config:
        n_embd: int = 500
        bias: bool = True
        dropout: float = 0.1
    return MLP(Config())

@pytest.fixture
def sample_module_layer_norm():
    """Fixture to create a sample LayerNorm model and return it."""
    return LayerNorm(ndim=500, use_bias=True)

def test_mlp_forward_pass(sample_module_mlp):
    """Simple test to ensure forward pass runs and outputs the right shape."""
    x = torch.randn(64, 500)
    model = sample_module_mlp
    y = model(x)
    assert y.size() == (64, 500), (
        f"Output has shape {y.size()}, but expected {(64, 500)}"
    )

def test_layer_norm_forward_pass(sample_module_layer_norm):
    """Simple test to ensure forward pass runs and outputs the right shape."""
    x = torch.randn(64, 500)
    model = sample_module_layer_norm
    y = model(x)
    assert y.size() == (64, 500), (
        f"Output has shape {y.size()}, but expected {(64, 500)}"
    )

def test_mlp_runtime(sample_module_mlp):
    x = torch.randn(64, 500).cuda()
    model = sample_module_mlp
    model.cuda()

    def time_forward_pass():
        start_time = time.time()
        y = model(x)
        end_time = time.time()
        elapsed = end_time - start_time
        return elapsed

    print("")
    for i in range(5):
        elapsed = time_forward_pass()  
        print(f"Forward pass (MLP) run {i} took", elapsed*1000, "milliseconds.")

    max_allowed_time = 1.0
    assert elapsed < max_allowed_time, (
        f"Forward pass took {elapsed:.4f}s, which is longer than the allowed {max_allowed_time}s."
    )

def test_layer_norm_runtime(sample_module_layer_norm):
    x = torch.randn(64, 500).cuda()
    model = sample_module_layer_norm
    model.cuda()

    def time_forward_pass():
        start_time = time.time()
        y = model(x)
        end_time = time.time()
        elapsed = end_time - start_time
        return elapsed

    print("")
    for i in range(5):
        elapsed = time_forward_pass()  
        print(f"Forward pass (LayerNorm) run {i} took", elapsed*1000, "milliseconds.")

    max_allowed_time = 1.0
    assert elapsed < max_allowed_time, (
        f"Forward pass took {elapsed:.4f}s, which is longer than the allowed {max_allowed_time}s."
    )


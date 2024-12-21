

import time
import pytest
import torch
from cxt.kv_cache import KVCache

@pytest.fixture
def kv_cache():
    """Fixture that returns a default KVCache for testing."""
    device = "cpu"
    max_batch_size = 2
    max_seq_length = 8
    n_head = 2
    head_size = 4
    return KVCache(
        max_batch_size=max_batch_size,
        max_seq_length=max_seq_length,
        n_head=n_head,
        head_size=head_size,
        device=device
    )

def test_kvcache_init(kv_cache):
    """Test that KVCache can be instantiated and shapes are correct."""
    # Check shapes
    assert kv_cache.src_key_cache.shape == (2, 2, 8, 4)
    assert kv_cache.src_value_cache.shape == (2, 2, 8, 4)
    assert kv_cache.tgt_key_cache.shape == (2, 2, 8, 4)
    assert kv_cache.tgt_value_cache.shape == (2, 2, 8, 4)
    
    # Check initial lengths
    assert kv_cache.src_len == 0
    assert kv_cache.tgt_len == 0

def test_kvcache_update_source(kv_cache):
    """Test updating source key/value caches."""
    # Create fake source key/value data
    # Shape: (batch_size=2, n_head=2, seq_len=3, head_size=4)
    key_src = torch.randn(2, 2, 3, 4)
    val_src = torch.randn(2, 2, 3, 4)
    
    # Update source cache
    kv_cache.update_source(key_src, val_src)
    
    # Ensure lengths and stored data match
    assert kv_cache.src_len == 3
    
    # Check the first 3 tokens match what we set
    torch.testing.assert_close(kv_cache.src_key_cache[:, :, :3, :], key_src)
    torch.testing.assert_close(kv_cache.src_value_cache[:, :, :3, :], val_src)

def test_kvcache_update_target(kv_cache):
    """Test updating target key/value caches at a given position."""
    key_tgt = torch.randn(2, 2, 1, 4)   # seq_len=1
    val_tgt = torch.randn(2, 2, 1, 4)
    
    position = 0
    kv_cache.update_target(key_tgt, val_tgt, position)
    
    # Tgt_len should now be 1
    assert kv_cache.tgt_len == 1
    
    torch.testing.assert_close(kv_cache.tgt_key_cache[:, :, 0:1, :], key_tgt)
    torch.testing.assert_close(kv_cache.tgt_value_cache[:, :, 0:1, :], val_tgt)

def test_kvcache_get_kv(kv_cache):
    """Test concatenation of source and target KV."""
    # 1) Update source with 2 tokens
    src_k = torch.randn(2, 2, 2, 4)
    src_v = torch.randn(2, 2, 2, 4)
    kv_cache.update_source(src_k, src_v)
    
    # 2) Update target with 1 token at position 0
    tgt_k = torch.randn(2, 2, 1, 4)
    tgt_v = torch.randn(2, 2, 1, 4)
    kv_cache.update_target(tgt_k, tgt_v, position=0)
    
    # 3) Retrieve concatenated K,V
    #    position=0 => we want up to index 0 in the target
    full_k, full_v = kv_cache.get_kv(position=0)
    
    # full_k and full_v should have shape: (batch_size=2, n_head=2, seq_len=3, head_size=4)
    expected_seq_len = kv_cache.src_len + 1  # (2 + 1) = 3
    assert full_k.shape == (2, 2, expected_seq_len, 4)
    assert full_v.shape == (2, 2, expected_seq_len, 4)
    
    # Manually check the first 2 tokens (from src) and the last 1 token (from tgt)
    torch.testing.assert_close(full_k[:, :, :2, :], src_k)
    torch.testing.assert_close(full_k[:, :, 2:3, :], tgt_k)
    torch.testing.assert_close(full_v[:, :, :2, :], src_v)
    torch.testing.assert_close(full_v[:, :, 2:3, :], tgt_v)

def test_kvcache_clear(kv_cache):
    """Test clearing the cache sets lengths to 0 and zeros out memory."""
    # Update with something
    key = torch.ones(2, 2, 1, 4)
    val = torch.ones(2, 2, 1, 4)
    kv_cache.update_source(key, val) 
    kv_cache.update_target(key, val, position=0)

    # Clear
    kv_cache.clear()
    
    # After clearing, src_len & tgt_len should be 0
    assert kv_cache.src_len == 0
    assert kv_cache.tgt_len == 0
    
    # The caches should be all zeros
    assert torch.count_nonzero(kv_cache.src_key_cache) == 0
    assert torch.count_nonzero(kv_cache.src_value_cache) == 0
    assert torch.count_nonzero(kv_cache.tgt_key_cache) == 0
    assert torch.count_nonzero(kv_cache.tgt_value_cache) == 0


def test_kvcache_update_efficiency(kv_cache):
    """
    Test the speed of updating source and target repeatedly.
    If the updates take too long, the test will fail.
    """
    # Let's simulate a slightly larger scenario than in the basic tests:
    # We'll create random data and measure how long many updates take.
    # Adjust these parameters to stress-test performance.
    batch_size = kv_cache.max_batch_size
    n_head =  kv_cache.src_key_cache.shape[1]
    head_size = kv_cache.src_key_cache.shape[3]
    
    # We'll run multiple updates across a sequence.
    seq_len = kv_cache.max_seq_length
    total_updates = seq_len  # One update per token

    # Warm-up (optionally do one forward to "warm" caches, JIT compilers, etc.)
    dummy_key = torch.randn(batch_size, n_head, 1, head_size)
    dummy_val = torch.randn(batch_size, n_head, 1, head_size)
    kv_cache.update_target(dummy_key, dummy_val, position=0)
    kv_cache.clear()  # Reset

    start_time = time.time()
    for i in range(total_updates):
        # Each update: one token
        key = torch.randn(batch_size, n_head, 1, head_size)
        val = torch.randn(batch_size, n_head, 1, head_size)
        kv_cache.update_target(key, val, position=i)
    end_time = time.time()

    elapsed_time = end_time - start_time
    # For example, expect all updates to finish in under 0.1s for these shapes
    assert elapsed_time < 0.1, (
        f"KVCache update took too long: {elapsed_time:.5f}s "
        f"(expected < 0.1s for {total_updates} updates)."
    )

def test_kvcache_get_kv_efficiency(kv_cache):
    """
    Test the speed of concatenating KV across many positions.
    """
    batch_size = kv_cache.max_batch_size
    n_head = kv_cache.src_key_cache.shape[1]
    head_size = kv_cache.src_key_cache.shape[3]
    seq_len = kv_cache.max_seq_length
    
    # Pre-populate with entire sequence
    key_src = torch.randn(batch_size, n_head, seq_len, head_size)
    val_src = torch.randn(batch_size, n_head, seq_len, head_size)
    kv_cache.update_source(key_src, val_src)
    
    # Now time repeated calls to get_kv
    start_time = time.time()
    for i in range(seq_len):
        _ = kv_cache.get_kv(position=i)  # We don't need the return
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    # Arbitrary threshold, adjust as appropriate
    assert elapsed_time < 0.1, (
        f"KVCache get_kv took too long: {elapsed_time:.5f}s "
        f"(expected < 0.1s for {seq_len} calls)."
    )

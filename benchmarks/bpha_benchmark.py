"""
Benchmark: BPHA Attention

Tests BPHA operator correctness and performance.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import numpy as np
import torch
import torch.nn.functional as F
from bpha import BPHAOperator, bpha_forward


def standard_attention(query, keys, values, scale=1.0):
    """Standard attention for comparison."""
    scores = torch.matmul(query, keys.transpose(-2, -1)) * scale
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, values)


def test_correctness():
    """Test BPHA against standard attention."""
    print("=" * 70)
    print("BPHA Correctness Test")
    print("=" * 70)

    batch_size = 2
    q_len = 1
    seq_len = 64
    hidden_dim = 64
    block_size = 16

    print(f"\nConfiguration:")
    print(f"  batch_size={batch_size}, q_len={q_len}, seq_len={seq_len}")
    print(f"  hidden_dim={hidden_dim}, block_size={block_size}")

    query = torch.randn(batch_size, q_len, hidden_dim)
    keys = torch.randn(batch_size, seq_len, hidden_dim)
    values = torch.randn(batch_size, seq_len, hidden_dim)

    num_blocks = (seq_len + block_size - 1) // block_size
    kv_blocks = []
    block_offsets = []

    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, seq_len)
        k_block = keys[:, start:end, :]
        v_block = values[:, start:end, :]
        kv_blocks.append((k_block, v_block))
        block_offsets.append(start)

    print(f"\nCreated {len(kv_blocks)} KV blocks")

    scale = 1.0 / (hidden_dim ** 0.5)

    expected = standard_attention(query, keys, values, scale)

    bpha_op = BPHAOperator(hidden_dim=hidden_dim, num_heads=1, block_size=block_size)
    result = bpha_op.forward(query, kv_blocks, block_offsets)

    result_func = bpha_forward(query, kv_blocks, block_offsets, scale)

    diff_op = torch.max(torch.abs(expected - result)).item()
    diff_func = torch.max(torch.abs(expected - result_func)).item()

    print(f"\nResults:")
    print(f"  Max diff (BPHAOperator): {diff_op:.2e}")
    print(f"  Max diff (bpha_forward): {diff_func:.2e}")

    if diff_op < 1e-4 and diff_func < 1e-4:
        print("  ✓ BPHA output matches standard attention!")
    else:
        print("  ✗ Warning: Large difference detected")


def benchmark_throughput():
    """Benchmark BPHA throughput."""
    print("\n" + "=" * 70)
    print("BPHA Throughput Benchmark")
    print("=" * 70)

    configs = [
        (1, 1, 128, 64, 16),
        (1, 1, 512, 64, 16),
        (1, 1, 1024, 64, 16),
        (4, 1, 512, 64, 16),
    ]

    print(f"\n{'Config':<25} {'Time':<12} {'Throughput':<15}")
    print("-" * 70)

    for batch_size, q_len, seq_len, hidden_dim, block_size in configs:
        query = torch.randn(batch_size, q_len, hidden_dim)

        num_blocks = (seq_len + block_size - 1) // block_size
        kv_blocks = []
        block_offsets = []

        for i in range(num_blocks):
            start = i * block_size
            end = min(start + block_size, seq_len)
            k = torch.randn(batch_size, end - start, hidden_dim)
            v = torch.randn(batch_size, end - start, hidden_dim)
            kv_blocks.append((k, v))
            block_offsets.append(start)

        bpha_op = BPHAOperator(hidden_dim=hidden_dim, block_size=block_size)

        num_iters = 100
        start = time.time()
        for _ in range(num_iters):
            _ = bpha_op.forward(query, kv_blocks, block_offsets)
        elapsed = time.time() - start

        throughput = num_iters / elapsed

        config_str = f"b={batch_size},q={q_len},s={seq_len},h={hidden_dim}"
        print(f"{config_str:<25} {elapsed/num_iters*1000:>8.3f}ms {throughput:>10.1f} iter/s")


if __name__ == "__main__":
    test_correctness()
    benchmark_throughput()
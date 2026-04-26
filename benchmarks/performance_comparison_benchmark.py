"""
Benchmark: BPHA vs Standard Attention Performance Comparison

Compares latency, throughput, and memory usage between BPHA and standard attention.
Phase 3 of performance comparison.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import torch
import torch.nn.functional as F
from typing import Tuple, List
from bpha import BPHAOperator
from pagedAttention import BlockTable


def standard_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    scale: float
) -> torch.Tensor:
    """
    Reference implementation of standard attention.

    Args:
        query: [batch, q_len, d_k]
        keys: [batch, kv_len, d_k]
        values: [batch, kv_len, d_v]
        scale: Attention scale (typically 1/sqrt(d_k))

    Returns:
        Attention output [batch, q_len, d_v]
    """
    scores = torch.matmul(query, keys.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, values)


def create_bpha_inputs(
    batch_size: int,
    q_len: int,
    seq_len: int,
    hidden_dim: int,
    block_size: int,
    device: torch.device
) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], List[int]]:
    """
    Create inputs for BPHA operator.

    Returns:
        query, kv_blocks, block_offsets
    """
    query = torch.randn(batch_size, q_len, hidden_dim, device=device)

    num_blocks = (seq_len + block_size - 1) // block_size
    kv_blocks = []
    block_offsets = []

    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, seq_len)
        k = torch.randn(batch_size, end - start, hidden_dim, device=device)
        v = torch.randn(batch_size, end - start, hidden_dim, device=device)
        kv_blocks.append((k, v))
        block_offsets.append(start)

    return query, kv_blocks, block_offsets


def create_standard_inputs(
    batch_size: int,
    q_len: int,
    seq_len: int,
    hidden_dim: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create inputs for standard attention.

    Returns:
        query, keys, values
    """
    query = torch.randn(batch_size, q_len, hidden_dim, device=device)
    keys = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    values = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    return query, keys, values


def benchmark_latency_comparison():
    """
    Compare latency between BPHA and standard attention with multiple configs.
    """
    print("=" * 80)
    print("LATENCY COMPARISON: BPHA vs Standard Attention")
    print("=" * 80)

    # Configs: (batch_size, q_len, seq_len, hidden_dim, block_size)
    configs = [
        (1, 1, 128, 64, 16),
        (1, 1, 128, 64, 64),
        (1, 1, 128, 64, 128),
        (1, 1, 512, 64, 16),
        (1, 1, 512, 64, 64),
        (1, 1, 512, 64, 128),
        (1, 1, 1024, 64, 16),
        (1, 1, 1024, 64, 64),
        (1, 1, 1024, 64, 128),
        (4, 1, 512, 64, 128),
        (4, 1, 1024, 64, 128),
        (1, 1, 512, 128, 128),
        (1, 1, 512, 256, 128),
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_warmup = 10
    num_iters = 100

    print(f"\nDevice: {device}")
    print(f"Warmup iterations: {num_warmup}")
    print(f"Measured iterations: {num_iters}")
    print()
    print(f"{'Config':<35} {'BPHA (ms)':<12} {'Standard (ms)':<15} {'Speedup':<10}")
    print("-" * 80)

    results = []
    for batch_size, q_len, seq_len, hidden_dim, block_size in configs:
        # Setup BPHA
        query_bpha, kv_blocks, block_offsets = create_bpha_inputs(
            batch_size, q_len, seq_len, hidden_dim, block_size, device
        )
        bpha_op = BPHAOperator(hidden_dim=hidden_dim, block_size=block_size).to(device)

        # Setup standard
        query_std, keys, values = create_standard_inputs(
            batch_size, q_len, seq_len, hidden_dim, device
        )
        scale = 1.0 / (hidden_dim ** 0.5)

        # Warmup BPHA
        for _ in range(num_warmup):
            _ = bpha_op.forward(query_bpha, kv_blocks, block_offsets)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Benchmark BPHA
        start = time.time()
        for _ in range(num_iters):
            _ = bpha_op.forward(query_bpha, kv_blocks, block_offsets)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        bpha_time = (time.time() - start) / num_iters * 1000

        # Warmup standard
        for _ in range(num_warmup):
            _ = standard_attention(query_std, keys, values, scale)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Benchmark standard
        start = time.time()
        for _ in range(num_iters):
            _ = standard_attention(query_std, keys, values, scale)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        std_time = (time.time() - start) / num_iters * 1000

        speedup = std_time / bpha_time if bpha_time > 0 else 0
        config_str = f"b={batch_size},q={q_len},s={seq_len},h={hidden_dim},bs={block_size}"
        print(f"{config_str:<35} {bpha_time:>8.3f}     {std_time:>8.3f}         {speedup:>6.2f}x")

        results.append({
            'config': config_str,
            'batch_size': batch_size,
            'q_len': q_len,
            'seq_len': seq_len,
            'hidden_dim': hidden_dim,
            'block_size': block_size,
            'bpha_time_ms': bpha_time,
            'standard_time_ms': std_time,
            'speedup': speedup,
        })

    return results


def benchmark_throughput_comparison():
    """
    Compare throughput between BPHA and standard attention.
    """
    print("\n" + "=" * 80)
    print("THROUGHPUT COMPARISON: BPHA vs Standard Attention")
    print("=" * 80)

    # Configs: (batch_size, q_len, seq_len, hidden_dim, block_size)
    configs = [
        (1, 1, 512, 64, 128),
        (4, 1, 512, 64, 128),
        (8, 1, 512, 64, 128),
        (1, 4, 512, 64, 128),
        (1, 1, 1024, 64, 128),
        (4, 1, 1024, 64, 128),
        (1, 1, 512, 128, 128),
        (1, 1, 512, 256, 128),
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_iters = 100

    print(f"\nDevice: {device}")
    print()
    print(f"{'Config':<30} {'BPHA tok/s':<15} {'Standard tok/s':<18} {'Ratio':<10}")
    print("-" * 80)

    results = []
    for batch_size, q_len, seq_len, hidden_dim, block_size in configs:
        # Setup BPHA
        query_bpha, kv_blocks, block_offsets = create_bpha_inputs(
            batch_size, q_len, seq_len, hidden_dim, block_size, device
        )
        bpha_op = BPHAOperator(hidden_dim=hidden_dim, block_size=block_size).to(device)

        # Setup standard
        query_std, keys, values = create_standard_inputs(
            batch_size, q_len, seq_len, hidden_dim, device
        )
        scale = 1.0 / (hidden_dim ** 0.5)

        # Benchmark BPHA
        start = time.time()
        for _ in range(num_iters):
            _ = bpha_op.forward(query_bpha, kv_blocks, block_offsets)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        bpha_time = time.time() - start
        bpha_throughput = (batch_size * q_len * num_iters) / bpha_time

        # Benchmark standard
        start = time.time()
        for _ in range(num_iters):
            _ = standard_attention(query_std, keys, values, scale)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        std_time = time.time() - start
        std_throughput = (batch_size * q_len * num_iters) / std_time

        ratio = bpha_throughput / std_throughput if std_throughput > 0 else 0
        config_str = f"b={batch_size},q={q_len},s={seq_len},h={hidden_dim}"
        print(f"{config_str:<30} {bpha_throughput:>10.0f}     {std_throughput:>10.0f}           {ratio:>6.2f}x")

        results.append({
            'config': config_str,
            'batch_size': batch_size,
            'q_len': q_len,
            'seq_len': seq_len,
            'hidden_dim': hidden_dim,
            'bpha_throughput': bpha_throughput,
            'standard_throughput': std_throughput,
            'ratio': ratio,
        })

    return results


def benchmark_memory_comparison():
    """
    Compare GPU memory usage between BPHA and standard attention.
    """
    print("\n" + "=" * 80)
    print("MEMORY COMPARISON: BPHA vs Standard Attention")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA not available - skipping memory benchmark")
        return []

    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Configs: (batch_size, seq_len, hidden_dim, block_size)
    configs = [
        (1, 512, 64, 128),
        (4, 512, 64, 128),
        (1, 1024, 64, 128),
        (4, 1024, 64, 128),
        (1, 2048, 64, 128),
        (1, 512, 128, 128),
        (1, 512, 256, 128),
    ]

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
    print(f"{'Config':<30} {'BPHA Peak MB':<15} {'Standard Peak MB':<18} {'Difference':<12}")
    print("-" * 80)

    results = []
    for batch_size, seq_len, hidden_dim, block_size in configs:
        q_len = 1

        # BPHA memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        query_bpha, kv_blocks, block_offsets = create_bpha_inputs(
            batch_size, q_len, seq_len, hidden_dim, block_size, device
        )
        bpha_op = BPHAOperator(hidden_dim=hidden_dim, block_size=block_size).to(device)
        _ = bpha_op.forward(query_bpha, kv_blocks, block_offsets)
        torch.cuda.synchronize()
        bpha_mem = torch.cuda.max_memory_allocated() / 1024**2

        del query_bpha, kv_blocks, bpha_op
        torch.cuda.empty_cache()

        # Standard memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        query_std, keys, values = create_standard_inputs(
            batch_size, q_len, seq_len, hidden_dim, device
        )
        scale = 1.0 / (hidden_dim ** 0.5)
        _ = standard_attention(query_std, keys, values, scale)
        torch.cuda.synchronize()
        std_mem = torch.cuda.max_memory_allocated() / 1024**2

        del query_std, keys, values
        torch.cuda.empty_cache()

        diff = bpha_mem - std_mem
        diff_pct = (diff / std_mem) * 100 if std_mem > 0 else 0
        config_str = f"b={batch_size},s={seq_len},h={hidden_dim}"
        print(f"{config_str:<30} {bpha_mem:>8.2f}       {std_mem:>8.2f}           {diff:>+6.2f} MB ({diff_pct:>+.1f}%)")

        results.append({
            'config': config_str,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'hidden_dim': hidden_dim,
            'bpha_mem_mb': bpha_mem,
            'standard_mem_mb': std_mem,
            'difference_mb': diff,
            'difference_pct': diff_pct,
        })

    return results


def benchmark_block_size_impact():
    """
    Analyze the impact of block size on BPHA performance.
    """
    print("\n" + "=" * 80)
    print("BLOCK SIZE IMPACT ANALYSIS")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 1
    q_len = 1
    seq_len = 512
    hidden_dim = 64

    block_sizes = [16, 32, 64, 128, 256]
    num_iters = 100

    print(f"\nFixed config: batch={batch_size}, q_len={q_len}, seq_len={seq_len}, hidden_dim={hidden_dim}")
    print(f"\n{'Block Size':<15} {'Latency (ms)':<15} {'Throughput (tok/s)':<20} {'Memory (MB)':<15}")
    print("-" * 80)

    results = []
    for block_size in block_sizes:
        torch.cuda.reset_peak_memory_stats() if device.type == 'cuda' else None
        torch.cuda.empty_cache() if device.type == 'cuda' else None

        query, kv_blocks, block_offsets = create_bpha_inputs(
            batch_size, q_len, seq_len, hidden_dim, block_size, device
        )
        bpha_op = BPHAOperator(hidden_dim=hidden_dim, block_size=block_size).to(device)

        # Warmup
        for _ in range(10):
            _ = bpha_op.forward(query, kv_blocks, block_offsets)

        # Benchmark
        start = time.time()
        for _ in range(num_iters):
            _ = bpha_op.forward(query, kv_blocks, block_offsets)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - start

        latency_ms = (elapsed / num_iters) * 1000
        throughput = (batch_size * q_len * num_iters) / elapsed

        mem_mb = 0
        if device.type == 'cuda':
            mem_mb = torch.cuda.max_memory_allocated() / 1024**2

        print(f"{block_size:<15} {latency_ms:>8.3f}       {throughput:>10.0f}           {mem_mb:>8.2f}")

        results.append({
            'block_size': block_size,
            'latency_ms': latency_ms,
            'throughput': throughput,
            'memory_mb': mem_mb,
        })

    return results


def main():
    """Run all benchmarks."""
    print("=" * 80)
    print("BPHA vs STANDARD ATTENTION PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("Running on CPU (CUDA not available)")

    # Run all benchmarks
    latency_results = benchmark_latency_comparison()
    throughput_results = benchmark_throughput_comparison()
    memory_results = benchmark_memory_comparison()
    block_size_results = benchmark_block_size_impact()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if latency_results:
        avg_speedup = sum(r['speedup'] for r in latency_results) / len(latency_results)
        print(f"\nAverage Latency Speedup (BPHA vs Standard): {avg_speedup:.2f}x")

    if throughput_results:
        avg_ratio = sum(r['ratio'] for r in throughput_results) / len(throughput_results)
        print(f"Average Throughput Ratio (BPHA/Standard): {avg_ratio:.2f}x")

    if memory_results:
        avg_mem_diff = sum(r['difference_mb'] for r in memory_results) / len(memory_results)
        print(f"Average Memory Difference: {avg_mem_diff:+.2f} MB")

    print("\n" + "=" * 80)
    print("Benchmark completed successfully!")
    print("=" * 80)

    return {
        'latency': latency_results,
        'throughput': throughput_results,
        'memory': memory_results,
        'block_size': block_size_results,
    }


if __name__ == "__main__":
    main()
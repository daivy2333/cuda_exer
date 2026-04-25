"""
Benchmark: Paged Attention Performance on Single GPU

Measures throughput and latency for BPHA on single 4060 (8GB).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import torch
from bpha import BPHAOperator, bpha_forward
from pagedAttention import BlockTable


def benchmark_throughput_detailed():
    """Detailed throughput benchmark with various configurations."""
    print("=" * 70)
    print("BPHA Throughput Benchmark (Single GPU)")
    print("=" * 70)

    configs = [
        (1, 1, 128, 64, 128),
        (1, 1, 256, 64, 128),
        (1, 1, 512, 64, 128),
        (1, 1, 1024, 64, 128),
        (1, 1, 2048, 64, 128),
        (4, 1, 512, 64, 128),
        (8, 1, 512, 64, 128),
        (1, 4, 512, 64, 128),
        (1, 1, 512, 128, 128),
        (1, 1, 512, 256, 128),
    ]

    print(f"\n{'Config':<30} {'Time':<12} {'Throughput':<15} {'Tokens/sec':<15}")
    print("-" * 70)

    results = []
    for batch_size, q_len, seq_len, hidden_dim, block_size in configs:
        query = torch.randn(batch_size, q_len, hidden_dim).cuda()

        num_blocks = (seq_len + block_size - 1) // block_size
        kv_blocks = []
        block_offsets = []

        for i in range(num_blocks):
            start = i * block_size
            end = min(start + block_size, seq_len)
            k = torch.randn(batch_size, end - start, hidden_dim).cuda()
            v = torch.randn(batch_size, end - start, hidden_dim).cuda()
            kv_blocks.append((k, v))
            block_offsets.append(start)

        bpha_op = BPHAOperator(hidden_dim=hidden_dim, block_size=block_size).cuda()

        num_iters = 100
        start = time.time()
        for _ in range(num_iters):
            _ = bpha_op.forward(query, kv_blocks, block_offsets)
        elapsed = time.time() - start

        throughput = num_iters / elapsed
        tokens_per_sec = (batch_size * q_len * num_iters) / elapsed

        config_str = f"b={batch_size},q={q_len},s={seq_len},h={hidden_dim}"
        print(f"{config_str:<30} {elapsed/num_iters*1000:>8.3f}ms {throughput:>10.1f} iter/s {tokens_per_sec:>12.0f}")

        results.append({
            'batch_size': batch_size,
            'q_len': q_len,
            'seq_len': seq_len,
            'hidden_dim': hidden_dim,
            'time_ms': elapsed / num_iters * 1000,
            'throughput': throughput,
            'tokens_per_sec': tokens_per_sec,
        })

    return results


def benchmark_latency():
    """Measure latency percentiles."""
    print("\n" + "=" * 70)
    print("BPHA Latency Benchmark")
    print("=" * 70)

    batch_size, q_len, seq_len, hidden_dim = 1, 1, 512, 64
    block_size = 128

    query = torch.randn(batch_size, q_len, hidden_dim).cuda()

    num_blocks = (seq_len + block_size - 1) // block_size
    kv_blocks = []
    block_offsets = []

    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, seq_len)
        k = torch.randn(batch_size, end - start, hidden_dim).cuda()
        v = torch.randn(batch_size, end - start, hidden_dim).cuda()
        kv_blocks.append((k, v))
        block_offsets.append(start)

    bpha_op = BPHAOperator(hidden_dim=hidden_dim, block_size=block_size).cuda()

    warmup = 10
    for _ in range(warmup):
        _ = bpha_op.forward(query, kv_blocks, block_offsets)

    latencies = []
    num_iters = 200
    for _ in range(num_iters):
        start = time.time()
        _ = bpha_op.forward(query, kv_blocks, block_offsets)
        latencies.append((time.time() - start) * 1000)

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]

    print(f"\nSequence length: {seq_len}, Hidden dim: {hidden_dim}")
    print(f"Warmup iterations: {warmup}")
    print(f"Measured iterations: {num_iters}")
    print(f"\nLatency percentiles:")
    print(f"  P50: {p50:.3f}ms")
    print(f"  P95: {p95:.3f}ms")
    print(f"  P99: {p99:.3f}ms")

    return {'p50': p50, 'p95': p95, 'p99': p99}


def benchmark_memory_efficiency():
    print("\n" + "=" * 70)
    print("Memory Efficiency Benchmark")
    print("=" * 70)

    block_size = 128

    scenarios = [
        ("Short sequences", [50, 60, 45, 55, 70]),
        ("Medium sequences", [200, 250, 300, 280, 220]),
        ("Mixed sequences", [50, 200, 100, 500, 80, 300, 150]),
        ("Long sequences", [500, 800, 1000, 1200]),
    ]

    print(f"\nBlock size: {block_size}")
    print("-" * 70)

    for scenario_name, seq_lengths in scenarios:
        print(f"\nScenario: {scenario_name}")
        print(f"  Sequence lengths: {seq_lengths}")

        total_tokens = sum(seq_lengths)

        num_blocks_needed = sum((l + block_size - 1) // block_size for l in seq_lengths)
        total_capacity = num_blocks_needed * block_size
        waste = total_capacity - total_tokens

        frag = waste / total_capacity if total_capacity > 0 else 0
        print(f"  Total tokens: {total_tokens}")
        print(f"  Blocks needed: {num_blocks_needed}")
        print(f"  Memory waste: {frag:.2%} ({waste} tokens)")
        print(f"  Memory utilization: {1-frag:.2%}")


def benchmark_gpu_memory():
    """Check GPU memory usage."""
    print("\n" + "=" * 70)
    print("GPU Memory Status")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    mem_allocated = torch.cuda.memory_allocated() / 1024**2
    mem_reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"Current allocated: {mem_allocated:.1f} MB")
    print(f"Current reserved: {mem_reserved:.1f} MB")

    batch_size, q_len, seq_len, hidden_dim = 4, 1, 1024, 64
    block_size = 128

    query = torch.randn(batch_size, q_len, hidden_dim).cuda()

    num_blocks = (seq_len + block_size - 1) // block_size
    kv_blocks = []
    block_offsets = []

    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, seq_len)
        k = torch.randn(batch_size, end - start, hidden_dim).cuda()
        v = torch.randn(batch_size, end - start, hidden_dim).cuda()
        kv_blocks.append((k, v))
        block_offsets.append(start)

    bpha_op = BPHAOperator(hidden_dim=hidden_dim, block_size=block_size).cuda()
    _ = bpha_op.forward(query, kv_blocks, block_offsets)

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Peak allocated during forward: {peak_mem:.1f} MB")


if __name__ == "__main__":
    benchmark_throughput_detailed()
    benchmark_latency()
    benchmark_memory_efficiency()
    benchmark_gpu_memory()
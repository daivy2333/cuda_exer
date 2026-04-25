"""
Benchmark: Memory Fragmentation Comparison

Compares memory fragmentation between contiguous and paged allocation.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import time
import numpy as np
from pagedAttention import BlockTable, PagedMemoryManager


def simulate_contiguous_allocation(num_sequences, seq_lengths, block_size=16):
    """
    Simulate traditional contiguous allocation.
    Returns memory waste due to fragmentation.
    """
    total_waste = 0
    total_capacity = 0

    for length in seq_lengths[:num_sequences]:
        capacity = ((length + block_size - 1) // block_size) * block_size
        waste = capacity - length
        total_waste += waste
        total_capacity += capacity

    return total_waste / total_capacity if total_capacity > 0 else 0


def benchmark_fragmentation():
    """Benchmark memory fragmentation for various sequence distributions."""
    print("=" * 70)
    print("Memory Fragmentation Benchmark")
    print("=" * 70)

    scenarios = [
        ("Short sequences", [50, 60, 45, 55, 70]),
        ("Medium sequences", [200, 250, 300, 280, 220]),
        ("Mixed sequences", [50, 200, 100, 500, 80, 300, 150]),
        ("Long sequences", [1000, 1200, 800, 1500, 1100]),
    ]

    block_size = 16
    num_blocks = 100

    print(f"\nBlock size: {block_size}")
    print("-" * 70)

    results = []

    for scenario_name, seq_lengths in scenarios:
        print(f"\nScenario: {scenario_name}")
        print(f"  Sequence lengths: {seq_lengths}")

        contiguous_frag = simulate_contiguous_allocation(
            len(seq_lengths), seq_lengths, block_size
        )
        print(f"  Contiguous fragmentation: {contiguous_frag:.2%}")

        bt = BlockTable(block_size=block_size, num_blocks=num_blocks)
        total_tokens = 0
        total_capacity = 0

        for i, length in enumerate(seq_lengths):
            if not bt.can_allocate((length + block_size - 1) // block_size):
                print(f"  Warning: Not enough blocks for sequence {i}")
                continue

            bt.allocate(seq_id=i, num_tokens=length)
            num_blocks_used = (length + block_size - 1) // block_size
            capacity = num_blocks_used * block_size
            total_tokens += length
            total_capacity += capacity

        paged_frag = 1 - (total_tokens / total_capacity) if total_capacity > 0 else 0
        print(f"  Paged fragmentation: {paged_frag:.2%}")

        improvement = (
            (contiguous_frag - paged_frag) / contiguous_frag
            if contiguous_frag > 0
            else 0
        )
        print(f"  Improvement: {improvement:.1%}")

        results.append(
            {
                "scenario": scenario_name,
                "contiguous_frag": contiguous_frag,
                "paged_frag": paged_frag,
                "improvement": improvement,
            }
        )

    print("\n" + "=" * 70)
    print("Summary Table")
    print("=" * 70)
    print(f"{'Scenario':<20} {'Contiguous':<12} {'Paged':<12} {'Improvement':<12}")
    print("-" * 70)

    for r in results:
        print(
            f"{r['scenario']:<20} {r['contiguous_frag']:>10.2%} {r['paged_frag']:>10.2%} {r['improvement']:>10.1%}"
        )


def benchmark_alloc_free(num_iterations=1000):
    """Benchmark allocation and deallocation speed."""
    print("\n" + "=" * 70)
    print("Allocation/Deallocation Speed Benchmark")
    print("=" * 70)

    block_size = 16
    num_blocks = 100
    seq_length = 100

    bt = BlockTable(block_size=block_size, num_blocks=num_blocks)

    print(f"\nConfiguration: block_size={block_size}, num_blocks={num_blocks}")
    print(f"Running {num_iterations} iterations...")

    start = time.time()
    for i in range(num_iterations):
        seq_id = i % 10
        try:
            bt.free(seq_id)
        except RuntimeError:
            pass  # Sequence may not exist yet, which is expected
        bt.allocate(seq_id=seq_id, num_tokens=seq_length)

    elapsed = time.time() - start
    ops_per_sec = (num_iterations * 2) / elapsed

    print(f"\nResults:")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Operations/sec: {ops_per_sec:.0f}")
    print(f"  Avg time per alloc+free: {elapsed / num_iterations * 1000:.4f}ms")


if __name__ == "__main__":
    benchmark_fragmentation()
    benchmark_alloc_free()

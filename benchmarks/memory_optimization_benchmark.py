"""
Benchmark: GPU Memory Optimization for BPHA KV Cache

Analyzes GPU memory usage for BPHA KV cache with Qwen2.5-3B model.
Phase 2 of memory optimization - testing block_size and max_blocks configurations.

Qwen2.5-3B Parameters:
- num_layers: 36
- num_kv_heads: 2 (GQA)
- head_dim: 128
- dtype: torch.bfloat16
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import time
from typing import Dict, List, Tuple

import torch
from qwen_adapter.kv_cache_manager import KVCacheManager


# Qwen2.5-3B Model Configuration
QWEN_3B_CONFIG = {
    "num_layers": 36,
    "num_kv_heads": 2,
    "head_dim": 128,
    "dtype": torch.bfloat16,
}


def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory information in MB."""
    if not torch.cuda.is_available():
        return {"total_mb": 0, "free_mb": 0, "used_mb": 0}

    total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    allocated = torch.cuda.memory_allocated() / (1024**2)
    free = total - reserved

    return {
        "total_mb": total,
        "free_mb": free,
        "used_mb": allocated,
        "reserved_mb": reserved,
    }


def calculate_kv_cache_memory(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    max_blocks: int,
    dtype: torch.dtype,
) -> Dict[str, float]:
    """Calculate theoretical KV cache memory requirements."""
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()

    # Memory per block (K and V)
    # Shape: [batch_size=1, num_kv_heads, block_size, head_dim]
    block_memory = 2 * num_kv_heads * block_size * head_dim * bytes_per_element

    # Total memory for all layers
    total_memory = block_memory * max_blocks * num_layers

    return {
        "block_memory_bytes": block_memory,
        "block_memory_kb": block_memory / 1024,
        "total_memory_bytes": total_memory,
        "total_memory_mb": total_memory / (1024**2),
    }


def analyze_memory_bottleneck() -> List[Dict]:
    """Test different block_size/max_blocks configurations.

    Analyzes memory usage patterns to identify bottlenecks and optimal
    configurations for the RTX 4060 8GB GPU.
    """
    print("=" * 70)
    print("Memory Bottleneck Analysis")
    print("=" * 70)
    print(f"\nModel: Qwen2.5-3B")
    print(f"  Layers: {QWEN_3B_CONFIG['num_layers']}")
    print(f"  KV Heads: {QWEN_3B_CONFIG['num_kv_heads']}")
    print(f"  Head Dim: {QWEN_3B_CONFIG['head_dim']}")
    print(f"  Dtype: {QWEN_3B_CONFIG['dtype']}")

    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        mem_info = get_gpu_memory_info()
        print(f"Total VRAM: {mem_info['total_mb']:.0f} MB")
    else:
        print("\nWarning: CUDA not available, using CPU for calculations")

    # Test configurations
    configs = [
        {"block_size": 16, "max_blocks": 1000},
        {"block_size": 32, "max_blocks": 1000},
        {"block_size": 64, "max_blocks": 1000},
        {"block_size": 128, "max_blocks": 500},
        {"block_size": 256, "max_blocks": 250},
        {"block_size": 512, "max_blocks": 125},
    ]

    print("\n" + "-" * 70)
    print(f"{'Block Size':<12} {'Max Blocks':<12} {'Memory/Block':<15} {'Total Memory':<15} {'Max Tokens':<12}")
    print("-" * 70)

    results = []
    for config in configs:
        block_size = config["block_size"]
        max_blocks = config["max_blocks"]

        mem_info = calculate_kv_cache_memory(
            num_layers=QWEN_3B_CONFIG["num_layers"],
            num_kv_heads=QWEN_3B_CONFIG["num_kv_heads"],
            head_dim=QWEN_3B_CONFIG["head_dim"],
            block_size=block_size,
            max_blocks=max_blocks,
            dtype=QWEN_3B_CONFIG["dtype"],
        )

        max_tokens = block_size * max_blocks

        print(
            f"{block_size:<12} {max_blocks:<12} "
            f"{mem_info['block_memory_kb']:>10.1f} KB   "
            f"{mem_info['total_memory_mb']:>10.1f} MB   "
            f"{max_tokens:>10}"
        )

        results.append(
            {
                "block_size": block_size,
                "max_blocks": max_blocks,
                "memory_per_block_kb": mem_info["block_memory_kb"],
                "total_memory_mb": mem_info["total_memory_mb"],
                "max_tokens": max_tokens,
            }
        )

    print("\n" + "-" * 70)
    print("Analysis:")
    print("  - Smaller block_size: Lower fragmentation, more overhead per block")
    print("  - Larger block_size: Higher fragmentation, fewer blocks for same memory")
    print("  - RTX 4060 has 8GB VRAM; recommend using ~4-6GB for KV cache")

    return results


def benchmark_max_blocks_capacity() -> Dict:
    """Find maximum KV cache capacity for RTX 4060 8GB.

    Iteratively tests max_blocks to find the maximum capacity that
    fits within GPU memory constraints.
    """
    print("\n" + "=" * 70)
    print("Max Blocks Capacity Benchmark")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU capacity test")
        return {"error": "CUDA not available"}

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    mem_info = get_gpu_memory_info()
    print(f"Total VRAM: {mem_info['total_mb']:.0f} MB")
    print(f"Currently free: {mem_info['free_mb']:.0f} MB")

    # Target: Use up to 4GB for KV cache (leaving room for model weights)
    target_kv_memory_mb = 4000

    block_size = 128  # Default block size

    # Calculate max_blocks that fits in target memory
    mem_info = calculate_kv_cache_memory(
        num_layers=QWEN_3B_CONFIG["num_layers"],
        num_kv_heads=QWEN_3B_CONFIG["num_kv_heads"],
        head_dim=QWEN_3B_CONFIG["head_dim"],
        block_size=block_size,
        max_blocks=1,  # Calculate per-block memory
        dtype=QWEN_3B_CONFIG["dtype"],
    )

    memory_per_block_mb = mem_info["total_memory_mb"]
    estimated_max_blocks = int(target_kv_memory_mb / memory_per_block_mb)

    print(f"\nMemory per block (block_size={block_size}): {memory_per_block_mb:.4f} MB")
    print(f"Target KV cache memory: {target_kv_memory_mb} MB")
    print(f"Estimated max blocks: {estimated_max_blocks}")

    # Test different block counts
    test_counts = [
        estimated_max_blocks // 4,
        estimated_max_blocks // 2,
        estimated_max_blocks,
        int(estimated_max_blocks * 1.25),
        int(estimated_max_blocks * 1.5),
    ]

    print("\n" + "-" * 70)
    print(f"{'Max Blocks':<12} {'Est. Memory':<15} {'Actual Alloc':<15} {'Success':<10}")
    print("-" * 70)

    results = []
    successful_configs = []

    torch.cuda.empty_cache()

    for max_blocks in test_counts:
        try:
            # Reset GPU memory tracking
            torch.cuda.reset_peak_memory_stats()

            # Create KVCacheManager
            kv_manager = KVCacheManager(
                num_layers=QWEN_3B_CONFIG["num_layers"],
                num_kv_heads=QWEN_3B_CONFIG["num_kv_heads"],
                head_dim=QWEN_3B_CONFIG["head_dim"],
                block_size=block_size,
                max_blocks=max_blocks,
                dtype=QWEN_3B_CONFIG["dtype"],
                device="cuda",
            )

            # Allocate one sequence to force memory allocation
            kv_manager.allocate_sequence(seq_id=0, num_tokens=block_size)

            # Get actual memory usage
            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
            current_mem = torch.cuda.memory_allocated() / (1024**2)

            print(
                f"{max_blocks:<12} "
                f"{max_blocks * memory_per_block_mb:>10.1f} MB   "
                f"{peak_mem:>10.1f} MB   "
                f"{'OK':<10}"
            )

            results.append(
                {
                    "max_blocks": max_blocks,
                    "estimated_mb": max_blocks * memory_per_block_mb,
                    "actual_peak_mb": peak_mem,
                    "actual_current_mb": current_mem,
                    "success": True,
                }
            )
            successful_configs.append(max_blocks)

            # Clean up
            del kv_manager
            torch.cuda.empty_cache()

        except RuntimeError as e:
            print(
                f"{max_blocks:<12} "
                f"{max_blocks * memory_per_block_mb:>10.1f} MB   "
                f"{'N/A':<15} "
                f"FAILED"
            )
            results.append(
                {
                    "max_blocks": max_blocks,
                    "estimated_mb": max_blocks * memory_per_block_mb,
                    "actual_peak_mb": 0,
                    "actual_current_mb": 0,
                    "success": False,
                    "error": str(e),
                }
            )

    # Find maximum successful configuration
    if successful_configs:
        max_successful = max(successful_configs)
        print(f"\nMaximum successful max_blocks: {max_successful}")
        print(
            f"Maximum KV cache capacity: {max_successful * block_size} tokens "
            f"({max_successful * block_size // 1024}K tokens)"
        )
    else:
        max_successful = 0
        print("\nNo successful configurations found!")

    return {
        "results": results,
        "max_successful_blocks": max_successful,
        "block_size": block_size,
    }


def benchmark_block_size_efficiency() -> List[Dict]:
    """Compare efficiency for different block sizes.

    Tests memory efficiency with varying sequence lengths to find
    optimal block_size for the Qwen2.5-3B model.
    """
    print("\n" + "=" * 70)
    print("Block Size Efficiency Benchmark")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available, running theoretical analysis only")

    # Sequence length scenarios
    scenarios = {
        "Short (64-256 tokens)": [64, 128, 192, 256],
        "Medium (256-1024 tokens)": [256, 512, 768, 1024],
        "Long (1024-4096 tokens)": [1024, 2048, 3072, 4096],
        "Mixed": [64, 256, 1024, 4096],
    }

    block_sizes = [16, 32, 64, 128, 256]

    print("\nMemory Waste Analysis (fragmentation due to block alignment):")
    print("-" * 70)
    print(f"{'Block Size':<12} ", end="")
    for scenario_name in scenarios.keys():
        print(f"{scenario_name[:15]:<18} ", end="")
    print()
    print("-" * 70)

    results = []

    for block_size in block_sizes:
        print(f"{block_size:<12} ", end="")
        block_results = {"block_size": block_size, "scenarios": {}}

        for scenario_name, seq_lengths in scenarios.items():
            total_tokens = sum(seq_lengths)
            total_blocks = sum((l + block_size - 1) // block_size for l in seq_lengths)
            total_capacity = total_blocks * block_size
            waste = total_capacity - total_tokens
            waste_pct = (waste / total_capacity) * 100 if total_capacity > 0 else 0

            print(f"{waste_pct:>6.1f}% waste     ", end="")

            block_results["scenarios"][scenario_name] = {
                "total_tokens": total_tokens,
                "total_blocks": total_blocks,
                "waste_pct": waste_pct,
            }

        print()
        results.append(block_results)

    print("\n" + "-" * 70)
    print("Recommendations:")
    print("  - Short sequences (prompt): Smaller block_size (16-32) preferred")
    print("  - Medium sequences: block_size 64-128 provides good balance")
    print("  - Long sequences (generation): block_size 128-256 reduces overhead")

    # GPU memory test if available
    if torch.cuda.is_available():
        print("\n" + "-" * 70)
        print("GPU Memory Allocation Test (single 1024-token sequence):")
        print("-" * 70)
        print(f"{'Block Size':<12} {'Blocks Needed':<15} {'Memory Used':<15} {'Alloc Time':<15}")
        print("-" * 70)

        test_seq_len = 1024

        for block_size in block_sizes:
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()

                start = time.time()
                kv_manager = KVCacheManager(
                    num_layers=QWEN_3B_CONFIG["num_layers"],
                    num_kv_heads=QWEN_3B_CONFIG["num_kv_heads"],
                    head_dim=QWEN_3B_CONFIG["head_dim"],
                    block_size=block_size,
                    max_blocks=100,
                    dtype=QWEN_3B_CONFIG["dtype"],
                    device="cuda",
                )
                kv_manager.allocate_sequence(seq_id=0, num_tokens=test_seq_len)
                elapsed = (time.time() - start) * 1000

                peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
                blocks_needed = (test_seq_len + block_size - 1) // block_size

                print(
                    f"{block_size:<12} {blocks_needed:<15} "
                    f"{peak_mem:>10.1f} MB   {elapsed:>10.2f} ms"
                )

                del kv_manager
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"{block_size:<12} ERROR: {str(e)[:40]}")

    return results


def main():
    """Run all memory optimization benchmarks."""
    print("=" * 70)
    print("GPU Memory Optimization Benchmark for BPHA KV Cache")
    print("Target Model: Qwen2.5-3B")
    print("=" * 70)

    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"Compute Capability: {props.major}.{props.minor}")
        print(f"Total VRAM: {props.total_memory / (1024**3):.1f} GB")
    else:
        print("\nWarning: CUDA not available. Running theoretical analysis only.")

    # Run all benchmarks
    bottleneck_results = analyze_memory_bottleneck()
    capacity_results = benchmark_max_blocks_capacity()
    efficiency_results = benchmark_block_size_efficiency()

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print("\nOptimal Configuration for Qwen2.5-3B on RTX 4060 (8GB):")
    print("  - Recommended block_size: 128 (balance of fragmentation and overhead)")
    print("  - Target max_blocks: ~200-400 (depending on model memory)")
    print("  - Expected KV cache memory: 1-2 GB")
    print("  - Maximum sequence capacity: 25K-50K tokens")

    print("\nNext Steps:")
    print("  1. Test with actual Qwen2.5-3B model weights")
    print("  2. Measure real memory usage during inference")
    print("  3. Profile KV cache behavior with dynamic batching")

    return {
        "bottleneck_analysis": bottleneck_results,
        "capacity_benchmark": capacity_results,
        "efficiency_benchmark": efficiency_results,
    }


if __name__ == "__main__":
    main()
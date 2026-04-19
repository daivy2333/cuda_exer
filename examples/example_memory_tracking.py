"""
Example: Memory Tracking

Demonstrates memory usage tracking and analysis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from memory import MemoryTracker, BlockAllocator


def main():
    print("=" * 60)
    print("Memory Tracking Example")
    print("=" * 60)

    print("\n--- MemoryTracker Demo ---")
    tracker = MemoryTracker(name="kv_cache")
    print(f"Initialized: {tracker}")

    print("\nSimulating allocations...")
    allocations = [
        (1024, "sequence_1"),
        (2048, "sequence_2"),
        (512, "sequence_3"),
        (4096, "sequence_4"),
    ]

    for size, tag in allocations:
        tracker.allocate(size_bytes=size, tag=tag)
        print(f"  Allocated {size} bytes for {tag}")

    print(f"\nCurrent stats:")
    stats = tracker.get_current_stats()
    print(f"  Allocated: {stats.allocated_bytes} bytes")
    print(f"  Peak: {stats.peak_bytes} bytes")
    print(f"  Allocations: {stats.num_allocations}")

    print("\n--- Freeing some memory ---")
    tracker.free(size_bytes=2048, tag="sequence_2")
    print("  Freed 2048 bytes from sequence_2")

    updated_stats = tracker.get_current_stats()
    print(f"  Now allocated: {updated_stats.allocated_bytes} bytes")
    print(f"  Fragmentation: {updated_stats.fragmentation:.2%}")

    print("\n--- System Memory Info ---")
    sys_mem = tracker.get_system_memory()
    for key, value in sys_mem.items():
        print(f"  {key}: {value:.2f}")

    print("\n--- Summary Report ---")
    tracker.print_summary()

    print("\n--- BlockAllocator Demo ---")
    alloc = BlockAllocator(num_blocks=20, block_size=4096)
    print(f"Initialized: {alloc}")

    print("\nAllocating blocks...")
    blocks1 = alloc.allocate(5)
    print(f"  Allocated blocks: {blocks1}")
    print(f"  Utilization: {alloc.get_utilization():.1%}")
    print(f"  Fragmentation: {alloc.get_fragmentation():.1%}")

    blocks2 = alloc.allocate(3)
    print(f"  Allocated blocks: {blocks2}")
    print(f"  Free blocks: {alloc.get_num_free()}")

    print("\nFreeing first allocation...")
    alloc.free(blocks1)
    print(f"  Free blocks: {alloc.get_num_free()}")
    print(f"  Fragmentation: {alloc.get_fragmentation():.1%}")

    print("\nRe-allocating...")
    new_blocks = alloc.allocate(4)
    print(f"  New blocks: {new_blocks}")
    print(f"  Can allocate 6 more: {alloc.can_allocate(6)}")


if __name__ == "__main__":
    main()
"""
Example: Block Table Usage

Demonstrates basic BlockTable operations for KV Cache management.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pagedAttention import BlockTable


def main():
    print("=" * 60)
    print("Block Table Example")
    print("=" * 60)

    block_size = 16
    num_blocks = 10

    print(f"\nInitializing BlockTable(block_size={block_size}, num_blocks={num_blocks})")
    bt = BlockTable(block_size=block_size, num_blocks=num_blocks)
    print(f"BlockTable: {bt}")

    print("\n--- Allocating sequences ---")

    seq1_blocks = bt.allocate(seq_id=1, num_tokens=50)
    print(f"Allocated seq_id=1 with {len(seq1_blocks)} blocks: {seq1_blocks}")
    print(f"  Free blocks remaining: {bt.get_num_free_blocks()}")

    seq2_blocks = bt.allocate(seq_id=2, num_tokens=30)
    print(f"Allocated seq_id=2 with {len(seq2_blocks)} blocks: {seq2_blocks}")
    print(f"  Free blocks remaining: {bt.get_num_free_blocks()}")

    print("\n--- Block Utilization ---")
    print(f"Used blocks: {bt.get_num_used_blocks()}/{bt.num_blocks}")
    print(f"Utilization: {bt.get_utilization():.1%}")

    print("\n--- Fragmentation Analysis ---")
    frag1 = bt.get_fragmentation_rate(seq_id=1)
    frag2 = bt.get_fragmentation_rate(seq_id=2)
    print(f"Seq 1 fragmentation: {frag1:.2%}")
    print(f"Seq 2 fragmentation: {frag2:.2%}")

    print("\n--- Freeing sequences ---")
    bt.free(seq_id=1)
    print(f"Freed seq_id=1, free blocks: {bt.get_num_free_blocks()}")

    bt.free(seq_id=2)
    print(f"Freed seq_id=2, free blocks: {bt.get_num_free_blocks()}")

    print("\n--- Re-allocation (block reuse) ---")
    new_blocks = bt.allocate(seq_id=3, num_tokens=40)
    print(f"Re-allocated seq_id=3 with {len(new_blocks)} blocks: {new_blocks}")

    print("\n--- Summary ---")
    bt.print_summary()


if __name__ == "__main__":
    main()
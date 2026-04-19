"""
Example: Paged Attention

Demonstrates attention computation with paged KV Cache.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
from pagedAttention import BlockTable, PagedMemoryManager, PagedAttention


def create_mock_kv_data(block_table, seq_id, num_tokens, hidden_dim=64):
    """Create mock KV data and store in blocks."""
    block_ids = block_table.get_block_ids(seq_id)

    for i, block_id in enumerate(block_ids):
        block = block_table.physical_blocks[block_id]
        tokens_in_block = min(block_table.block_size, num_tokens - i * block_table.block_size)

        k_data = np.random.randn(tokens_in_block, hidden_dim).astype(np.float32)
        v_data = np.random.randn(tokens_in_block, hidden_dim).astype(np.float32)

        block.k_data = k_data
        block.v_data = v_data
        block.tokens = [0] * tokens_in_block


def main():
    print("=" * 60)
    print("Paged Attention Example")
    print("=" * 60)

    block_size = 16
    hidden_dim = 64
    num_tokens = 50

    print(f"\nInitializing components:")
    print(f"  block_size={block_size}, hidden_dim={hidden_dim}, num_tokens={num_tokens}")

    bt = BlockTable(block_size=block_size, num_blocks=100)
    print(f"  BlockTable: {bt}")

    print("\n--- Allocating sequence ---")
    block_ids = bt.allocate(seq_id=1, num_tokens=num_tokens)
    print(f"Allocated blocks: {block_ids}")

    print("\n--- Storing mock KV data ---")
    create_mock_kv_data(bt, seq_id=1, num_tokens=num_tokens, hidden_dim=hidden_dim)
    print("Created mock KV data in physical blocks")

    print("\n--- Computing attention ---")
    pa = PagedAttention(block_size=block_size)

    query = torch.randn(1, 1, hidden_dim)
    print(f"Query shape: {query.shape}")

    output, new_kv = pa.forward(query, bt, seq_id=1, num_tokens=num_tokens)
    print(f"Output shape: {output.shape}")
    print(f"New KV shape: {new_kv[0].shape}")

    print("\n--- Verifying correctness ---")
    max_diff, standard_out, paged_out = pa.compare_with_standard_attention(
        pa, bt, seq_id=1, num_tokens=num_tokens, d_k=hidden_dim
    )
    print(f"Max difference from standard attention: {max_diff:.2e}")

    if max_diff < 1e-4:
        print("✓ Paged Attention output matches standard attention!")
    else:
        print("✗ Warning: Large difference detected")

    print("\n--- Memory statistics ---")
    pmm = PagedMemoryManager(block_size=block_size, num_blocks=100, hidden_dim=hidden_dim)
    pmm.allocate_sequence(seq_id=1, num_tokens=num_tokens)
    stats = pmm.get_memory_stats()
    print(f"Memory stats: {stats}")


if __name__ == "__main__":
    main()
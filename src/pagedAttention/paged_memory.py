"""
Paged Memory Manager: Manages KV Cache with block-paging strategy
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from .block_table import BlockTable, Block


class PagedMemoryManager:
    """
    Manages KV Cache memory using paged allocation.

    Features:
    - Non-contiguous physical block storage
    - Block-level allocation and deallocation
    - Memory utilization tracking
    - Support for variable-length sequences
    """

    def __init__(
        self, block_size: int = 128, num_blocks: int = 100, hidden_dim: int = 64
    ):
        """
        Initialize PagedMemoryManager.

        Args:
            block_size: Tokens per block
            num_blocks: Maximum physical blocks
            hidden_dim: Hidden dimension for K/V vectors
        """
        self.block_size = block_size
        self.hidden_dim = hidden_dim
        self.block_table = BlockTable(block_size=block_size, num_blocks=num_blocks)
        self.active_sequences: Dict[int, int] = {}

    def allocate_sequence(self, seq_id: int, num_tokens: int) -> bool:
        """
        Allocate memory for a new sequence.

        Args:
            seq_id: Unique sequence identifier
            num_tokens: Expected sequence length

        Returns:
            True if allocation successful
        """
        if seq_id in self.active_sequences:
            return False

        try:
            block_ids = self.block_table.allocate(seq_id, num_tokens)
            self.active_sequences[seq_id] = num_tokens

            for block_id in block_ids:
                block = self.block_table.physical_blocks[block_id]
                block.k_data = np.zeros((self.block_size, self.hidden_dim))
                block.v_data = np.zeros((self.block_size, self.hidden_dim))

            return True
        except RuntimeError:
            return False

    def free_sequence(self, seq_id: int):
        """Free memory allocated to a sequence."""
        self.block_table.free(seq_id)
        if seq_id in self.active_sequences:
            del self.active_sequences[seq_id]

    def append_tokens(
        self, seq_id: int, k_vectors: np.ndarray, v_vectors: np.ndarray
    ) -> bool:
        """
        Append new token KV vectors to sequence.

        Args:
            seq_id: Sequence identifier
            k_vectors: Key vectors [num_new_tokens, hidden_dim]
            v_vectors: Value vectors [num_new_tokens, hidden_dim]

        Returns:
            True if append successful
        """
        if seq_id not in self.active_sequences:
            return False

        num_new = k_vectors.shape[0]
        current_len = self.active_sequences[seq_id]
        block_ids = self.block_table.get_block_ids(seq_id)

        offset = 0
        remaining = num_new

        while remaining > 0:
            block_idx = (current_len + offset) // self.block_size
            if block_idx >= len(block_ids):
                return False

            block = self.block_table.physical_blocks[block_ids[block_idx]]
            pos_in_block = (current_len + offset) % self.block_size
            space_available = block.size - pos_in_block

            if block.k_data is None or block.v_data is None:
                return False

            copy_size = min(remaining, space_available)

            block.k_data[pos_in_block : pos_in_block + copy_size] = k_vectors[
                offset : offset + copy_size
            ]
            block.v_data[pos_in_block : pos_in_block + copy_size] = v_vectors[
                offset : offset + copy_size
            ]

            block.tokens.extend(range(copy_size))

            offset += copy_size
            remaining -= copy_size

        self.active_sequences[seq_id] += num_new
        return True

    def get_kv_blocks(self, seq_id: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get all KV blocks for a sequence.

        Returns:
            List of (K_block, V_block) tuples
        """
        blocks = self.block_table.get_physical_blocks(seq_id)
        result = []
        for block in blocks:
            if block.k_data is not None and block.v_data is not None:
                valid_tokens = len(block.tokens)
                if valid_tokens > 0:
                    result.append(
                        (block.k_data[:valid_tokens], block.v_data[:valid_tokens])
                    )
        return result

    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get memory utilization statistics.

        Returns:
            Dictionary with memory stats
        """
        total_blocks = self.block_table.num_blocks
        used_blocks = self.block_table.get_num_used_blocks()
        free_blocks = self.block_table.get_num_free_blocks()

        total_capacity = total_blocks * self.block_size * self.hidden_dim * 2
        block_size_bytes = 8
        total_memory = (
            total_blocks * self.block_size * self.hidden_dim * 2 * block_size_bytes
        )

        used_tokens = sum(self.active_sequences.values())
        utilized_capacity = used_tokens * self.hidden_dim * 2 * block_size_bytes

        return {
            "total_blocks": total_blocks,
            "used_blocks": used_blocks,
            "free_blocks": free_blocks,
            "utilization": used_blocks / total_blocks if total_blocks > 0 else 0,
            "total_memory_mb": total_memory / (1024 * 1024),
            "utilized_memory_mb": utilized_capacity / (1024 * 1024),
            "fragmentation_rate": 1.0 - (used_tokens / (used_blocks * self.block_size))
            if used_blocks > 0
            else 0,
            "active_sequences": len(self.active_sequences),
        }

    def reset(self):
        """Reset all allocations."""
        self.block_table.reset()
        self.active_sequences.clear()

    def can_allocate(self, num_tokens: int) -> bool:
        """Check if allocation is possible."""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        return num_blocks_needed <= self.block_table.get_num_free_blocks()

    def __repr__(self) -> str:
        stats = self.get_memory_stats()
        return (
            f"PagedMemoryManager(blocks={stats['used_blocks']}/{stats['total_blocks']}, "
            f"util={stats['utilization']:.1%})"
        )

"""
Block Table: Logical to Physical Block Mapping

Simulates OS page table for KV Cache block management.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Block:
    """Physical block storing KV pairs."""
    block_id: int
    size: int
    tokens: List[int] = field(default_factory=list)
    k_data: Optional[np.ndarray] = None
    v_data: Optional[np.ndarray] = None

    def is_full(self) -> bool:
        return len(self.tokens) >= self.size

    def remaining_capacity(self) -> int:
        return self.size - len(self.tokens)


class BlockTable:
    """
    Maps logical sequence blocks to physical memory blocks.

    Similar to OS page tables, this allows non-contiguous physical
    storage while maintaining logical sequence continuity.
    """

    def __init__(self, block_size: int = 16, num_blocks: int = 100):
        """
        Initialize BlockTable.

        Args:
            block_size: Number of tokens per block
            num_blocks: Maximum number of physical blocks
        """
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.physical_blocks: List[Block] = []
        self.mappings: Dict[int, List[int]] = {}
        self.free_blocks: List[int] = []
        self._initialize_blocks(num_blocks)

    def _initialize_blocks(self, num_blocks: int):
        """Pre-allocate physical blocks."""
        for i in range(num_blocks):
            self.physical_blocks.append(Block(block_id=i, size=self.block_size))
            self.free_blocks.append(i)

    def allocate(self, seq_id: int, num_tokens: int) -> List[int]:
        """
        Allocate blocks for a sequence.

        Args:
            seq_id: Sequence identifier
            num_tokens: Number of tokens in sequence

        Returns:
            List of physical block IDs allocated
        """
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        block_ids = []

        for i in range(num_blocks_needed):
            if not self.free_blocks:
                raise RuntimeError("No free blocks available")

            block_id = self.free_blocks.pop(0)
            block_ids.append(block_id)

        self.mappings[seq_id] = block_ids
        return block_ids

    def free(self, seq_id: int):
        """
        Free blocks allocated to a sequence.

        Args:
            seq_id: Sequence identifier
        """
        if seq_id not in self.mappings:
            return

        self.free_blocks.extend(self.mappings[seq_id])
        self.free_blocks.sort()
        del self.mappings[seq_id]

    def get_physical_blocks(self, seq_id: int) -> List[Block]:
        """
        Get physical blocks for a sequence.

        Args:
            seq_id: Sequence identifier

        Returns:
            List of physical blocks
        """
        if seq_id not in self.mappings:
            return []

        return [self.physical_blocks[bid] for bid in self.mappings[seq_id]]

    def get_block_ids(self, seq_id: int) -> List[int]:
        """Get physical block IDs for a sequence."""
        return self.mappings.get(seq_id, [])

    def get_fragmentation_rate(self, seq_id: int) -> float:
        """
        Calculate memory fragmentation rate for a sequence.

        Returns:
            Fragmentation rate (0.0 = no fragmentation, 1.0 = 100% waste)
        """
        if seq_id not in self.mappings:
            return 0.0

        block_ids = self.mappings[seq_id]
        if not block_ids:
            return 0.0

        last_block = self.physical_blocks[block_ids[-1]]
        tokens_in_last = len(last_block.tokens)
        wasted_in_last = last_block.size - tokens_in_last

        total_capacity = len(block_ids) * self.block_size
        if total_capacity == 0:
            return 0.0

        return wasted_in_last / total_capacity

    def get_utilization(self) -> float:
        """
        Calculate overall block utilization.

        Returns:
            Utilization rate (0.0 to 1.0)
        """
        used_blocks = sum(1 for bid in range(self.num_blocks)
                         if bid not in self.free_blocks)
        total_capacity = used_blocks * self.block_size

        if total_capacity == 0:
            return 0.0

        used_tokens = sum(len(block.tokens)
                          for block in self.physical_blocks
                          if block.block_id not in self.free_blocks)

        return used_tokens / total_capacity

    def get_num_free_blocks(self) -> int:
        """Return number of available blocks."""
        return len(self.free_blocks)

    def get_num_used_blocks(self) -> int:
        """Return number of allocated blocks."""
        return self.num_blocks - len(self.free_blocks)

    def reset(self):
        """Reset all allocations."""
        self.mappings.clear()
        self.free_blocks = list(range(self.num_blocks))
        for block in self.physical_blocks:
            block.tokens.clear()
            block.k_data = None
            block.v_data = None

    def __repr__(self) -> str:
        return (f"BlockTable(block_size={self.block_size}, "
                f"used={self.get_num_used_blocks()}/{self.num_blocks})")
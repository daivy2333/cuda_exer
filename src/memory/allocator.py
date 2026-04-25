"""
Block Allocator: Memory allocation for fixed-size blocks
"""

from collections import deque
from typing import List, Optional, Set
from dataclasses import dataclass


@dataclass
class Allocation:
    """Represents a memory allocation."""

    block_id: int
    size: int
    in_use: bool = True


class BlockAllocator:
    """
    Allocates memory from a pool of fixed-size blocks.

    Features:
    - O(1) allocation and deallocation
    - No external fragmentation
    - Efficient block reuse
    """

    def __init__(self, num_blocks: int, block_size: int):
        """
        Initialize BlockAllocator.

        Args:
            num_blocks: Total number of blocks
            block_size: Size of each block in bytes
        """
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.total_size = num_blocks * block_size

        self.free_blocks: deque = deque(range(num_blocks))
        self.allocations: List[Allocation] = [
            Allocation(block_id=i, size=block_size, in_use=False)
            for i in range(num_blocks)
        ]

    def allocate(self, num_blocks_requested: int = 1) -> Optional[List[int]]:
        """
        Allocate blocks from the pool.

        Args:
            num_blocks_requested: Number of consecutive blocks needed

        Returns:
            List of allocated block IDs, or None if not enough space
        """
        if num_blocks_requested > len(self.free_blocks):
            return None

        allocated = []
        for _ in range(num_blocks_requested):
            if not self.free_blocks:
                return None
            block_id = self.free_blocks.popleft()
            self.allocations[block_id].in_use = True
            allocated.append(block_id)

        return allocated

    def free(self, block_ids: List[int]):
        """
        Free allocated blocks.

        Args:
            block_ids: List of block IDs to free
        """
        for block_id in block_ids:
            if 0 <= block_id < self.num_blocks:
                self.allocations[block_id].in_use = False
                if block_id not in self.free_blocks:
                    self.free_blocks.append(block_id)
        self.free_blocks = deque(sorted(self.free_blocks, reverse=True))

    def free_all(self):
        """Free all allocated blocks."""
        for i in range(self.num_blocks):
            self.allocations[i].in_use = False
        self.free_blocks = deque(range(self.num_blocks))

    def get_num_free(self) -> int:
        """Get number of free blocks."""
        return len(self.free_blocks)

    def get_num_used(self) -> int:
        """Get number of used blocks."""
        return self.num_blocks - len(self.free_blocks)

    def get_utilization(self) -> float:
        """Get block utilization (0.0 to 1.0)."""
        return self.get_num_used() / self.num_blocks if self.num_blocks > 0 else 0.0

    def get_fragmentation(self) -> float:
        """
        Compute fragmentation due to non-consecutive free blocks.

        Returns:
            Fragmentation ratio (0.0 = no fragmentation, 1.0 = fully fragmented)
        """
        if not self.free_blocks:
            return 0.0

        if len(self.free_blocks) == self.num_blocks:
            return 0.0

        consecutive_groups = 1
        sorted_free = sorted(self.free_blocks)
        for i in range(1, len(sorted_free)):
            if sorted_free[i] != sorted_free[i - 1] + 1:
                consecutive_groups += 1

        max_groups = len(self.free_blocks)
        return 1.0 - (consecutive_groups / max_groups)

    def can_allocate(self, num_blocks: int) -> bool:
        """Check if allocation is possible."""
        return num_blocks <= len(self.free_blocks)

    def reset(self):
        """Reset allocator to initial state."""
        self.free_blocks = deque(range(self.num_blocks))
        for alloc in self.allocations:
            alloc.in_use = False

    def __repr__(self) -> str:
        return (
            f"BlockAllocator(blocks={self.get_num_used()}/{self.num_blocks}, "
            f"util={self.get_utilization():.1%}, frag={self.get_fragmentation():.1%})"
        )

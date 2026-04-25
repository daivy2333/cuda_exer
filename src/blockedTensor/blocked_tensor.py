"""
Blocked Tensor: Compiler-friendly blocked tensor representation
"""

from typing import Tuple, Optional, List
from dataclasses import dataclass
import itertools
import numpy as np


@dataclass
class LayoutConstraint:
    """Describes constraints on tensor layout for compiler optimization."""

    alignment: int = 16
    contiguity: str = "non-contiguous"
    access_pattern: str = "blocked-random"
    prefer_blocked: bool = True


class BlockedTensor:
    """
    Represents a tensor with explicit block structure for compiler optimization.

    The blocked tensor abstraction allows the compiler to:
    - Understand memory access patterns
    - Apply block-level loop transformations
    - Optimize prefetching and caching
    """

    def __init__(
        self,
        base_shape: Tuple[int, ...],
        block_size: Tuple[int, ...],
        block_map: Optional[List[int]] = None,
        data: Optional[np.ndarray] = None,
    ):
        """
        Initialize BlockedTensor.

        Args:
            base_shape: Logical tensor shape (seq_len, hidden_dim)
            block_size: Size of each block
            block_map: Mapping from logical blocks to physical storage
            data: Actual tensor data
        """
        self.base_shape = base_shape
        self.block_size = block_size
        self.block_map = block_map or list(range(self._num_blocks()))
        self.layout_constraint = LayoutConstraint()

        if data is not None:
            self.data = data
        else:
            self.data = np.zeros(self._total_elements(), dtype=np.float32)

    def _num_blocks(self) -> int:
        """Calculate number of blocks needed."""
        num_blocks = 1
        for dim, size in enumerate(self.base_shape):
            block_dim = self.block_size[dim]
            num_blocks *= (size + block_dim - 1) // block_dim
        return num_blocks

    @property
    def num_blocks(self) -> int:
        """Number of blocks in tensor."""
        return self._num_blocks()

    def _total_elements(self) -> int:
        """Total number of elements in tensor."""
        return int(np.prod(self.base_shape))

    def _block_shape(self) -> Tuple[int, ...]:
        """Shape of each block."""
        return self.block_size

    def get_block(self, block_idx: int) -> np.ndarray:
        """
        Get data for a specific block.

        Args:
            block_idx: Logical block index

        Returns:
            Block data as numpy array
        """
        physical_idx = (
            self.block_map[block_idx] if block_idx < len(self.block_map) else block_idx
        )

        block_elements = int(np.prod(self.block_size))
        start = physical_idx * block_elements
        end = start + block_elements

        flat_block = self.data[start:end]
        return flat_block.reshape(self.block_size)

    def set_block(self, block_idx: int, block_data: np.ndarray):
        """
        Set data for a specific block.

        Args:
            block_idx: Logical block index
            block_data: Data to store in block
        """
        physical_idx = (
            self.block_map[block_idx] if block_idx < len(self.block_map) else block_idx
        )

        block_elements = int(np.prod(self.block_size))
        start = physical_idx * block_elements
        end = start + block_elements

        flat_block = block_data.flatten()
        self.data[start:end] = flat_block[:block_elements]

    def get_logical_index(
        self, block_idx: int, within_block_idx: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """
        Convert block index to logical tensor index.

        Args:
            block_idx: Block index
            within_block_idx: Index within block

        Returns:
            Logical tensor index
        """
        num_dims = len(self.base_shape)
        logical_idx = []
        temp_block_idx = block_idx

        for dim in range(num_dims):
            blocks_per_dim = (
                self.base_shape[dim] + self.block_size[dim] - 1
            ) // self.block_size[dim]

            blocks_per_higher_dims = 1
            for higher_dim in range(dim + 1, num_dims):
                higher_blocks = (
                    self.base_shape[higher_dim] + self.block_size[higher_dim] - 1
                ) // self.block_size[higher_dim]
                blocks_per_higher_dims *= higher_blocks

            block_coord = (temp_block_idx // blocks_per_higher_dims) % blocks_per_dim
            logical_idx.append(
                block_coord * self.block_size[dim] + within_block_idx[dim]
            )

        return tuple(logical_idx)

    def physical_index(self, logical_index: Tuple[int, ...]) -> Tuple[int, int]:
        """
        Convert logical index to (physical_block_id, offset_within_block).

        Args:
            logical_index: Logical tensor index

        Returns:
            Tuple of (block_idx, offset_within_block)
        """
        block_idx = 0
        offset = 0
        num_dims = len(self.base_shape)
        block_strides = [1] * num_dims

        for dim in range(num_dims - 2, -1, -1):
            blocks_in_dim = (
                self.base_shape[dim + 1] + self.block_size[dim + 1] - 1
            ) // self.block_size[dim + 1]
            block_strides[dim] = block_strides[dim + 1] * blocks_in_dim

        for dim, idx in enumerate(logical_index):
            block_coord = idx // self.block_size[dim]
            offset_in_dim = idx % self.block_size[dim]

            block_idx += block_coord * block_strides[dim]

        offset_within_block = 0
        block_element_strides = [1] * num_dims
        for dim in range(num_dims - 2, -1, -1):
            block_element_strides[dim] = (
                block_element_strides[dim + 1] * self.block_size[dim + 1]
            )

        for dim, idx in enumerate(logical_index):
            offset_in_dim = idx % self.block_size[dim]
            offset_within_block += offset_in_dim * block_element_strides[dim]

        return block_idx, offset_within_block

    def get_layout_info(self) -> dict:
        """
        Get layout information for compiler.

        Returns:
            Dictionary with layout metadata
        """
        return {
            "base_shape": self.base_shape,
            "block_size": self.block_size,
            "num_blocks": self._num_blocks(),
            "total_elements": self._total_elements(),
            "constraint": {
                "alignment": self.layout_constraint.alignment,
                "contiguity": self.layout_constraint.contiguity,
                "access_pattern": self.layout_constraint.access_pattern,
                "prefer_blocked": self.layout_constraint.prefer_blocked,
            },
        }

    def __repr__(self) -> str:
        return (
            f"BlockedTensor(shape={self.base_shape}, "
            f"block_size={self.block_size}, "
            f"blocks={self._num_blocks()})"
        )


class BlockedTensorView:
    """
    View of a blocked tensor with sliced access patterns.
    """

    def __init__(
        self,
        tensor: BlockedTensor,
        start_idx: Tuple[int, ...],
        end_idx: Tuple[int, ...],
    ):
        """
        Create a view of blocked tensor.

        Args:
            tensor: Base blocked tensor
            start_idx: Start logical index (inclusive)
            end_idx: End logical index (exclusive)
        """
        self.tensor = tensor
        self.start_idx = start_idx
        self.end_idx = end_idx

    def get_block_coverage(self) -> List[int]:
        """Get list of blocks touched by this view."""
        covered_blocks = []
        for block_idx in range(self.tensor._num_blocks()):
            for within_idx in self._iter_within_block(block_idx):
                logical_idx = self.tensor.get_logical_index(block_idx, within_idx)
                if self._is_in_range(logical_idx):
                    if block_idx not in covered_blocks:
                        covered_blocks.append(block_idx)
                    break
        return covered_blocks

    def _is_in_range(self, logical_idx: Tuple[int, ...]) -> bool:
        """Check if logical index is within view range."""
        return all(
            s <= idx < e for s, idx, e in zip(self.start_idx, logical_idx, self.end_idx)
        )

    def _iter_within_block(self, block_idx: int):
        """Iterate over valid indices within a block."""
        block_shape = self.tensor.block_size

        for idx in itertools.product(*[range(s) for s in block_shape]):
            yield idx

    def __repr__(self) -> str:
        return f"BlockedTensorView({self.tensor}[{self.start_idx}:{self.end_idx}])"

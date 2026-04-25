"""
Blocked Tensor: Compiler-friendly blocked tensor representation
"""

from typing import Tuple, Optional, List
from dataclasses import dataclass
import itertools
import numpy as np


@dataclass
class LayoutConstraint:
    alignment: int = 128
    contiguity: str = "non-contiguous"
    access_pattern: str = "blocked-random"
    prefer_blocked: bool = True
    cache_line_size: int = 128
    shared_mem_size: int = 49152
    element_size_bytes: int = 4


class BlockedTensor:
    def __init__(
        self,
        base_shape: Tuple[int, ...],
        block_size: Tuple[int, ...],
        block_map: Optional[List[int]] = None,
        data: Optional[np.ndarray] = None,
    ):
        self.base_shape = base_shape
        self.block_size = block_size
        self.block_map = block_map or list(range(self._num_blocks()))
        self.layout_constraint = LayoutConstraint()

        if data is not None:
            self.data = data
        else:
            self.data = np.zeros(self._total_elements(), dtype=np.float32)

    def _num_blocks(self) -> int:
        num_blocks = 1
        for dim, size in enumerate(self.base_shape):
            block_dim = self.block_size[dim]
            num_blocks *= (size + block_dim - 1) // block_dim
        return num_blocks

    @property
    def num_blocks(self) -> int:
        return self._num_blocks()

    def _total_elements(self) -> int:
        return int(np.prod(self.base_shape))

    def _block_shape(self) -> Tuple[int, ...]:
        return self.block_size

    def get_block_strides(self) -> Tuple[int, ...]:
        if len(self.block_size) == 1:
            return (1,)
        strides = [1]
        for i in range(len(self.block_size) - 1):
            strides.append(strides[-1] * self.block_size[len(self.block_size) - 1 - i])
        return tuple(reversed(strides))

    def get_physical_adjacency(self, block_idx: int) -> List[int]:
        if block_idx >= len(self.block_map):
            return []
        physical_idx = self.block_map[block_idx]
        neighbors = []
        for i, mapped_idx in enumerate(self.block_map):
            if mapped_idx in [physical_idx - 1, physical_idx + 1]:
                neighbors.append(i)
        return neighbors

    def get_layout_info(self) -> dict:
        return {
            "base_shape": self.base_shape,
            "block_size": self.block_size,
            "num_blocks": self._num_blocks(),
            "total_elements": self._total_elements(),
            "block_strides": self.get_block_strides(),
            "constraint": {
                "alignment": self.layout_constraint.alignment,
                "contiguity": self.layout_constraint.contiguity,
                "access_pattern": self.layout_constraint.access_pattern,
                "prefer_blocked": self.layout_constraint.prefer_blocked,
                "cache_line_size": self.layout_constraint.cache_line_size,
                "shared_mem_size": self.layout_constraint.shared_mem_size,
                "element_size_bytes": self.layout_constraint.element_size_bytes,
            },
        }

    def __repr__(self) -> str:
        return (
            f"BlockedTensor(shape={self.base_shape}, "
            f"block_size={self.block_size}, "
            f"blocks={self._num_blocks()})"
        )

    def get_block(self, block_idx: int) -> np.ndarray:
        physical_idx = (
            self.block_map[block_idx] if block_idx < len(self.block_map) else block_idx
        )

        block_elements = int(np.prod(self.block_size))
        start = physical_idx * block_elements
        end = start + block_elements

        flat_block = self.data[start:end]
        return flat_block.reshape(self.block_size)

    def set_block(self, block_idx: int, block_data: np.ndarray):
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


class BlockedTensorView:
    def __init__(
        self,
        tensor: BlockedTensor,
        start_idx: Tuple[int, ...],
        end_idx: Tuple[int, ...],
    ):
        self.tensor = tensor
        self.start_idx = start_idx
        self.end_idx = end_idx

    def get_block_coverage(self) -> List[int]:
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
        return all(
            s <= idx < e for s, idx, e in zip(self.start_idx, logical_idx, self.end_idx)
        )

    def _iter_within_block(self, block_idx: int):
        block_shape = self.tensor.block_size

        for idx in itertools.product(*[range(s) for s in block_shape]):
            yield idx

    def __repr__(self) -> str:
        return f"BlockedTensorView({self.tensor}[{self.start_idx}:{self.end_idx}])"

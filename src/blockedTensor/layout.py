"""
Tensor Layout: Layout metadata for compiler optimization
"""

from typing import Tuple, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class ContiguityType(Enum):
    """Contiguity classification for tensor regions."""
    CONTIGUOUS = 'contiguous'
    NON_CONTIGUOUS = 'non-contiguous'
    PARTIALLY_CONTIGUOUS = 'partially-contiguous'
    BLOCKED = 'blocked'


class AccessPattern(Enum):
    """Access pattern classification."""
    SEQUENTIAL = 'sequential'
    RANDOM = 'random'
    BLOCKED_RANDOM = 'blocked-random'
    STRIDED = 'strided'
    BLOCKED_STRIDED = 'blocked-strided'


@dataclass
class TensorLayout:
    """
    Metadata describing tensor memory layout for compiler optimization.
    """
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    block_size: Optional[Tuple[int, ...]] = None
    contiguity: ContiguityType = ContiguityType.NON_CONTIGUOUS
    access_pattern: AccessPattern = AccessPattern.RANDOM
    alignment_bytes: int = 16
    preferred_tile_shape: Optional[Tuple[int, ...]] = None
    metadata: dict = field(default_factory=dict)

    def is_contiguous(self) -> bool:
        """Check if tensor is contiguous in memory."""
        return self.contiguity == ContiguityType.CONTIGUOUS

    def is_blocked(self) -> bool:
        """Check if tensor uses blocked layout."""
        return self.block_size is not None

    def get_effective_stride(self, dim: int) -> int:
        """
        Get effective stride for given dimension considering block layout.

        Args:
            dim: Dimension index

        Returns:
            Effective stride
        """
        if self.strides:
            return self.strides[dim]
        if self.block_size:
            return int(self.strides[-1]) if self.strides else 1
        return 1

    def estimate_cache_friendliness(self) -> float:
        """
        Estimate cache friendliness score (0.0 to 1.0).

        Returns:
            Score where 1.0 is most cache-friendly
        """
        if self.access_pattern == AccessPattern.SEQUENTIAL:
            return 1.0
        elif self.access_pattern == AccessPattern.BLOCKED_STRIDED:
            return 0.8
        elif self.access_pattern == AccessPattern.BLOCKED_RANDOM:
            return 0.6
        elif self.access_pattern == AccessPattern.STRIDED:
            return 0.4
        else:
            return 0.2

    def suggest_tile_shape(self, cache_size: int, element_size: int = 4) -> Tuple[int, ...]:
        """
        Suggest tile shape for given cache size.

        Args:
            cache_size: Cache size in bytes
            element_size: Size of each element in bytes

        Returns:
            Suggested tile shape
        """
        if self.preferred_tile_shape:
            return self.preferred_tile_shape

        elements_per_cache = cache_size // element_size
        tile_elements = int(elements_per_cache ** (1.0 / len(self.shape)))

        tile_shape = []
        for dim_size in self.shape:
            tile_dim = min(dim_size, max(1, tile_elements))
            tile_shape.append(tile_dim)

        return tuple(tile_shape)

    @classmethod
    def from_tensor(cls, tensor, block_size: Optional[Tuple[int, ...]] = None) -> 'TensorLayout':
        """
        Create TensorLayout from a tensor.

        Args:
            tensor: Input tensor (numpy or torch)
            block_size: Block size if applicable

        Returns:
            TensorLayout instance
        """
        import numpy as np
        import torch

        if isinstance(tensor, np.ndarray):
            shape = tensor.shape
            strides = tensor.strides
        elif isinstance(tensor, torch.Tensor):
            shape = tuple(tensor.shape)
            strides = tuple(tensor.stride())
        else:
            raise TypeError("Tensor must be numpy array or torch tensor")

        contiguity = cls._compute_contiguity(shape, strides)

        access_pattern = cls._infer_access_pattern(strides, shape)

        return cls(
            shape=shape,
            strides=strides,
            block_size=block_size,
            contiguity=contiguity,
            access_pattern=access_pattern
        )

    @staticmethod
    def _compute_contiguity(shape: Tuple[int, ...], strides: Tuple[int, ...]) -> ContiguityType:
        """Compute contiguity from shape and strides."""
        if len(shape) == 1:
            return ContiguityType.CONTIGUOUS

        expected_stride = 1
        is_contiguous = True
        for dim in range(len(shape) - 1, -1, -1):
            if shape[dim] == 1:
                continue
            if strides[dim] != expected_stride:
                is_contiguous = False
                break
            expected_stride *= shape[dim]

        if is_contiguous:
            return ContiguityType.CONTIGUOUS

        return ContiguityType.NON_CONTIGUOUS

    @staticmethod
    def _infer_access_pattern(strides: Tuple[int, ...], shape: Tuple[int, ...]) -> AccessPattern:
        """Infer access pattern from strides."""
        if len(strides) == 1:
            return AccessPattern.SEQUENTIAL

        expected_stride = 1
        for dim in range(len(shape) - 1, -1, -1):
            if shape[dim] == 1:
                continue
            if strides[dim] != expected_stride:
                if strides[dim] > expected_stride * shape[dim + 1] if dim + 1 < len(shape) else False:
                    return AccessPattern.STRIDED
                return AccessPattern.RANDOM
            expected_stride *= shape[dim]

        return AccessPattern.SEQUENTIAL

    def __repr__(self) -> str:
        return (f"TensorLayout(shape={self.shape}, "
                f"contiguity={self.contiguity.value}, "
                f"access_pattern={self.access_pattern.value})")
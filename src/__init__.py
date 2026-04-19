"""
CUDA Exercise: 2.5-D Tensor Parallelism with Paged Attention

A modular implementation for single-GPU environments.
"""

__version__ = '0.1.0'
__author__ = 'CUDA Exercise'

from .pagedAttention import BlockTable, PagedMemoryManager, PagedAttention
from .bpha import BPHAOperator, bpha_forward
from .dynamicBatching import AdaptiveBatcher, M1M1Queue
from .blockedTensor import BlockedTensor, TensorLayout
from .memory import MemoryTracker, BlockAllocator

__all__ = [
    'BlockTable',
    'PagedMemoryManager',
    'PagedAttention',
    'BPHAOperator',
    'bpha_forward',
    'AdaptiveBatcher',
    'M1M1Queue',
    'BlockedTensor',
    'TensorLayout',
    'MemoryTracker',
    'BlockAllocator',
]
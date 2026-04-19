"""
Block-Paged Hybrid Attention (BPHA)

Implements the hybrid attention mechanism with block-paged KV cache.
"""

from .bpha_operator import BPHAOperator
from .bpha_compute import bpha_forward, bpha_backward

__all__ = ['BPHAOperator', 'bpha_forward', 'bpha_backward']
__version__ = '0.1.0'
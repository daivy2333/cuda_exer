"""
Block-Paged Hybrid Attention (BPHA)

Implements the hybrid attention mechanism with block-paged KV cache.
"""

from .bpha_operator import BPHAOperator
from .bpha_compute import bpha_forward, bpha_backward

__all__ = ['BPHAOperator', 'bpha_forward', 'bpha_backward']
__version__ = '0.1.0'

# Export CUDA function if available
try:
    from .cuda import paged_attention_fused

    __all__.append("paged_attention_fused")
except ImportError:
    pass
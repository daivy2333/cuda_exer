"""
Blocked Tensor Abstraction

Compiler-friendly tensor representation with known block structure.
"""

from .blocked_tensor import BlockedTensor, BlockedTensorView
from .layout import TensorLayout, LayoutConstraint

__all__ = ['BlockedTensor', 'BlockedTensorView', 'TensorLayout', 'LayoutConstraint']
__version__ = '0.1.0'
"""
Dynamic Batching Module

Implements adaptive batch processing based on queuing theory.
"""

from .adaptive_batcher import AdaptiveBatcher
from .queue_model import M1M1Queue, QueueStats

__all__ = ['AdaptiveBatcher', 'M1M1Queue', 'QueueStats']
__version__ = '0.1.0'
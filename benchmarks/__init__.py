"""
Benchmarks package
"""

from .memory_benchmark import benchmark_fragmentation, benchmark_alloc_free
from .bpha_benchmark import test_correctness, benchmark_throughput
from .batching_benchmark import (
    test_queue_model,
    verify_theoretical_formulas,
    benchmark_adaptive_batching,
    simulate_burst_traffic
)

__all__ = [
    'benchmark_fragmentation',
    'benchmark_alloc_free',
    'test_correctness',
    'benchmark_throughput',
    'test_queue_model',
    'verify_theoretical_formulas',
    'benchmark_adaptive_batching',
    'simulate_burst_traffic',
]
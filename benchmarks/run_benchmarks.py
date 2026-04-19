"""
Run All Benchmarks
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from memory_benchmark import benchmark_fragmentation, benchmark_alloc_free
from bpha_benchmark import test_correctness, benchmark_throughput
from batching_benchmark import (
    test_queue_model,
    verify_theoretical_formulas,
    benchmark_adaptive_batching,
    simulate_burst_traffic
)


def main():
    print("\n" + "=" * 70)
    print("RUNNING ALL BENCHMARKS")
    print("=" * 70)

    benchmark_fragmentation()
    benchmark_alloc_free()
    test_correctness()
    benchmark_throughput()
    test_queue_model()
    verify_theoretical_formulas()
    benchmark_adaptive_batching()
    simulate_burst_traffic()

    print("\n" + "=" * 70)
    print("ALL BENCHMARKS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
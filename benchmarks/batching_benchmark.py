"""
Benchmark: Dynamic Batching Performance

Tests adaptive batching under various load conditions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import numpy as np
from dynamicBatching import AdaptiveBatcher, M1M1Queue


def test_queue_model():
    """Test M/M/1 queue model calculations."""
    print("=" * 70)
    print("M/M/1 Queue Model Verification")
    print("=" * 70)

    test_cases = [
        (10.0, 15.0, "Low load"),
        (14.0, 15.0, "High load"),
        (5.0, 10.0, "Moderate load"),
    ]

    print(f"\n{'Scenario':<15} {'λ':<8} {'μ':<8} {'ρ':<8} {'Lq':<10} {'W':<10} {'Stable':<8}")
    print("-" * 70)

    for lam, mu, name in test_cases:
        queue = M1M1Queue(arrival_rate=lam, service_rate=mu)
        stats = queue.get_stats()

        print(f"{name:<15} {lam:<8.2f} {mu:<8.2f} {stats.utilization:<8.2%} "
              f"{stats.avg_queue_length:<10.3f} {stats.avg_response_time:<10.3f} {str(stats.is_stable):<8}")


def verify_theoretical_formulas():
    """Verify M/M/1 theoretical formulas."""
    print("\n" + "=" * 70)
    print("Theoretical Formula Verification")
    print("=" * 70)

    lam = 8.0
    mu = 10.0
    rho = lam / mu

    queue = M1M1Queue(arrival_rate=lam, service_rate=mu)

    Lq_theoretical = (rho ** 2) / (1 - rho)
    Wq_theoretical = Lq_theoretical / lam
    W_theoretical = Wq_theoretical + (1 / mu)

    stats = queue.get_stats()

    print(f"\nConfiguration: λ={lam}, μ={mu}, ρ={rho:.2f}")
    print("-" * 50)
    print(f"{'Metric':<20} {'Theoretical':<15} {'Actual':<15} {'Diff':<12}")
    print("-" * 50)
    print(f"{'Lq (avg queue len)':<20} {Lq_theoretical:<15.4f} {stats.avg_queue_length:<15.4f} "
          f"{abs(Lq_theoretical - stats.avg_queue_length):<12.4f}")
    print(f"{'Wq (avg wait time)':<20} {Wq_theoretical:<15.4f} {stats.avg_waiting_time:<15.4f} "
          f"{abs(Wq_theoretical - stats.avg_waiting_time):<12.4f}")
    print(f"{'W (avg resp time)':<20} {W_theoretical:<15.4f} {stats.avg_response_time:<15.4f} "
          f"{abs(W_theoretical - stats.avg_response_time):<12.4f}")

    max_diff = max(
        abs(Lq_theoretical - stats.avg_queue_length),
        abs(Wq_theoretical - stats.avg_waiting_time),
        abs(W_theoretical - stats.avg_response_time)
    )

    if max_diff < 1e-6:
        print("\n✓ All theoretical formulas verified!")
    else:
        print(f"\n✗ Warning: Max difference = {max_diff:.2e}")


def benchmark_adaptive_batching():
    """Benchmark adaptive batching decisions."""
    print("\n" + "=" * 70)
    print("Adaptive Batching Benchmark")
    print("=" * 70)

    batcher = AdaptiveBatcher(max_batch_size=8, min_batch_size=1, latency_target=0.2)

    print(f"\nConfiguration: max_batch={batcher.max_batch_size}, latency_target={batcher.latency_target}s")

    loads = [
        (5.0, "Low"),
        (10.0, "Medium"),
        (14.0, "High"),
    ]

    print(f"\n{'Load':<10} {'ρ':<10} {'Batch Size':<12} {'Queue Len':<12} {'Decision':<20}")
    print("-" * 70)

    for lam, name in loads:
        mu = 15.0
        batcher.arrival_rate = lam
        batcher.service_rate = mu

        decision = batcher.decide_batch_size()

        print(f"{name:<10} {lam/mu:<10.2%} {decision.batch_size:<12} "
              f"{decision.queue_length:<12} {'high' if decision.batch_size == 8 else 'medium' if decision.batch_size == 4 else 'low':<20}")


def simulate_burst_traffic():
    """Simulate burst traffic patterns."""
    print("\n" + "=" * 70)
    print("Burst Traffic Simulation")
    print("=" * 70)

    batcher = AdaptiveBatcher(max_batch_size=8, latency_target=0.1)

    print("\nSimulating burst arrivals...")

    for burst_size in [2, 5, 10, 15, 20]:
        for i in range(burst_size):
            req = type('Request', (), {
                'request_id': i,
                'arrival_time': time.time(),
                'seq_length': 100 + (i % 100)
            })()
            batcher.add_request(req)

        decision = batcher.decide_batch_size()
        batch = batcher.get_batch()
        batcher.complete_batch(batch)

        print(f"  Burst {burst_size}: queue={decision.queue_length}, "
              f"ρ={decision.lambda_rate/decision.mu_rate:.2%}, "
              f"batch={len(batch)}")


if __name__ == "__main__":
    test_queue_model()
    verify_theoretical_formulas()
    benchmark_adaptive_batching()
    simulate_burst_traffic()
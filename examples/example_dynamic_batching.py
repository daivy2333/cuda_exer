"""
Example: Dynamic Batching

Demonstrates adaptive batch processing with M/M/1 queue model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
from dynamicBatching import AdaptiveBatcher, M1M1Queue


def simulate_requests(batcher, num_requests=20):
    """Simulate request arrivals."""
    from dynamicBatching.adaptive_batcher import Request
    print(f"\nSimulating {num_requests} request arrivals...")

    for i in range(num_requests):
        request = Request(
            request_id=i,
            arrival_time=time.time(),
            seq_length=100 + (i % 200),
            metadata={}
        )

        batcher.add_request(request)

        if i % 5 == 4:
            batch = batcher.get_batch()
            if batch:
                print(f"  Processed batch of {len(batch)} requests")
                batcher.complete_batch(batch)

        time.sleep(0.01)

    remaining = batcher.get_batch()
    while remaining:
        print(f"  Processed final batch of {len(remaining)} requests")
        batcher.complete_batch(remaining)
        remaining = batcher.get_batch()


def main():
    print("=" * 60)
    print("Dynamic Batching Example")
    print("=" * 60)

    print("\n--- M/M/1 Queue Model Demo ---")

    arrival_rate = 10.0
    service_rate = 15.0

    queue = M1M1Queue(arrival_rate=arrival_rate, service_rate=service_rate)
    print(f"Queue: {queue}")

    stats = queue.get_stats()
    print(f"\nQueue Statistics:")
    print(f"  Utilization (ρ): {stats.utilization:.2%}")
    print(f"  Avg queue length (Lq): {stats.avg_queue_length:.2f}")
    print(f"  Avg waiting time (Wq): {stats.avg_waiting_time:.3f}s")
    print(f"  Avg response time (W): {stats.avg_response_time:.3f}s")
    print(f"  Throughput: {stats.throughput:.2f} req/s")
    print(f"  System stable: {stats.is_stable}")

    print("\n--- Varying Load Conditions ---")

    for rho in [0.3, 0.5, 0.7, 0.9, 0.95]:
        mu = arrival_rate / rho
        q = M1M1Queue(arrival_rate=arrival_rate, service_rate=mu)
        stats = q.get_stats()
        print(f"ρ={rho:.2%}: Lq={stats.avg_queue_length:.2f}, W={stats.avg_response_time:.3f}s, stable={stats.is_stable}")

    print("\n--- Optimal Batch Size ---")

    batcher = AdaptiveBatcher(max_batch_size=8, min_batch_size=1, latency_target=0.2)
    print(f"AdaptiveBatcher: max_batch={batcher.max_batch_size}, latency_target={batcher.latency_target}s")

    simulate_requests(batcher, num_requests=20)

    print("\n--- Final Statistics ---")
    final_stats = batcher.get_stats()
    print(f"Total completed: {final_stats['total_completed']}")
    print(f"Final queue length: {final_stats['queue_length']}")
    print(f"Arrival rate: {final_stats['arrival_rate']:.2f} req/s")
    print(f"Service rate: {final_stats['service_rate']:.2f} req/s")


if __name__ == "__main__":
    main()
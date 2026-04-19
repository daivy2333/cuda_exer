"""
Adaptive Batcher: Dynamic batch size adjustment based on system load
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import time
import threading


@dataclass
class Request:
    """Represents a single inference request."""
    request_id: int
    arrival_time: float
    seq_length: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchDecision:
    """Result of batch size decision."""
    batch_size: int
    timestamp: float
    lambda_rate: float
    mu_rate: float
    queue_length: int


class AdaptiveBatcher:
    """
    Adaptive batch processor based on queuing theory.

    Uses M/M/1 queue model to dynamically adjust batch size:
    - High load: Increase batch size for throughput
    - Low load: Decrease batch size for latency
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        min_batch_size: int = 1,
        latency_target: float = 0.2,
        window_size: float = 1.0
    ):
        """
        Initialize AdaptiveBatcher.

        Args:
            max_batch_size: Maximum batch size
            min_batch_size: Minimum batch size
            latency_target: Target latency in seconds
            window_size: Time window for rate estimation
        """
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.latency_target = latency_target
        self.window_size = window_size

        self.request_queue: List[Request] = []
        self.arrival_times: List[float] = []
        self.service_times: List[float] = []
        self.completed_requests: List[Request] = []

        self.start_time = time.time()
        self.lock = threading.Lock()

    def add_request(self, request: Request):
        """Add a request to the queue."""
        with self.lock:
            self.request_queue.append(request)
            self.arrival_times.append(request.arrival_time)

    def get_batch(self) -> List[Request]:
        """
        Get next batch based on current system state.

        Returns:
            List of requests to process in next batch
        """
        with self.lock:
            decision = self.decide_batch_size()
            batch_size = decision.batch_size

            batch = self.request_queue[:batch_size]
            self.request_queue = self.request_queue[batch_size:]

            for req in batch:
                req.start_time = time.time()

            return batch

    def complete_batch(self, batch: List[Request]):
        """Record completion of a batch."""
        with self.lock:
            for req in batch:
                req.completion_time = time.time()
                service_time = req.completion_time - req.start_time
                self.service_times.append(service_time)
                self.completed_requests.append(req)

    def decide_batch_size(self) -> BatchDecision:
        """
        Decide optimal batch size using queuing theory.

        Returns:
            BatchDecision with recommended batch size
        """
        current_time = time.time()
        recent_arrivals = [t for t in self.arrival_times
                          if current_time - t <= self.window_size]

        lambda_rate = len(recent_arrivals) / self.window_size if self.window_size > 0 else 0

        recent_services = [t for t in self.service_times
                           if current_time - t <= self.window_size]
        mu_rate = len(recent_services) / sum(recent_services) if recent_services else 0

        if mu_rate == 0:
            mu_rate = 1.0

        rho = lambda_rate / mu_rate

        if rho > 0.8:
            batch_size = self.max_batch_size
        elif rho > 0.5:
            batch_size = (self.max_batch_size + self.min_batch_size) // 2
        else:
            batch_size = self.min_batch_size

        return BatchDecision(
            batch_size=batch_size,
            timestamp=current_time,
            lambda_rate=lambda_rate,
            mu_rate=mu_rate,
            queue_length=len(self.request_queue)
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current batching statistics.

        Returns:
            Dictionary with statistics
        """
        with self.lock:
            current_time = time.time()
            recent_arrivals = [t for t in self.arrival_times
                              if current_time - t <= self.window_size]
            recent_services = [t for t in self.service_times
                               if current_time - t <= self.window_size]

            lambda_rate = len(recent_arrivals) / self.window_size if self.window_size > 0 else 0
            mu_rate = len(recent_services) / sum(recent_services) if recent_services else 0

            return {
                'queue_length': len(self.request_queue),
                'total_completed': len(self.completed_requests),
                'arrival_rate': lambda_rate,
                'service_rate': mu_rate,
                'rho': lambda_rate / mu_rate if mu_rate > 0 else 0,
                'batch_decision': self.decide_batch_size().__dict__
            }

    def reset(self):
        """Reset all state."""
        with self.lock:
            self.request_queue.clear()
            self.arrival_times.clear()
            self.service_times.clear()
            self.completed_requests.clear()
            self.start_time = time.time()

    def simulate_batch_processing(self, num_requests: int, service_time: float = 0.05):
        """
        Simulate batch processing for testing.

        Args:
            num_requests: Number of requests to simulate
            service_time: Average service time per request
        """
        current_time = time.time()

        for i in range(num_requests):
            arrival_interval = abs(hash(str(i))) % 100 / 1000.0
            request = Request(
                request_id=i,
                arrival_time=current_time + i * arrival_interval,
                seq_length=128 + (i % 256)
            )
            self.add_request(request)

            time.sleep(0.01)

        while self.request_queue:
            batch = self.get_batch()
            if batch:
                time.sleep(service_time)
                self.complete_batch(batch)
"""
M/M/1 Queue Model for adaptive batching analysis
"""

from typing import Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class QueueStats:
    """Statistics from M/M/1 queue analysis."""
    utilization: float
    avg_queue_length: float
    avg_waiting_time: float
    avg_response_time: float
    throughput: float
    is_stable: bool


class M1M1Queue:
    """
    M/M/1 Queue model for inference serving analysis.

    Represents a single-server queue with:
    - Poisson arrival process (M)
    - Exponential service times (M)
    - Single server (1)
    """

    def __init__(self, arrival_rate: float = 0.0, service_rate: float = 1.0):
        """
        Initialize M/M/1 queue.

        Args:
            arrival_rate: Lambda (requests per second)
            service_rate: Mu (services per second)
        """
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate

    @property
    def utilization(self) -> float:
        """System utilization (rho = lambda / mu)."""
        if self.service_rate == 0:
            return 0.0
        return self.arrival_rate / self.service_rate

    @property
    def is_stable(self) -> bool:
        """Check if system is stable (rho < 1)."""
        return self.utilization < 1.0

    def avg_queue_length(self) -> float:
        """
        Calculate average queue length (L_q).

        L_q = rho^2 / (1 - rho)
        """
        rho = self.utilization
        if rho >= 1.0:
            return float('inf')
        if rho <= 0.0:
            return 0.0
        return (rho ** 2) / (1 - rho)

    def avg_number_in_system(self) -> float:
        """
        Calculate average number in system (L).

        L = rho / (1 - rho) = L_q + rho
        """
        rho = self.utilization
        if rho >= 1.0:
            return float('inf')
        if rho <= 0.0:
            return 0.0
        return rho / (1 - rho)

    def avg_waiting_time(self) -> float:
        """
        Calculate average waiting time in queue (W_q).

        W_q = L_q / lambda = rho / (mu - lambda)
        """
        if not self.is_stable:
            return float('inf')
        if self.arrival_rate <= 0.0:
            return 0.0
        return self.avg_queue_length() / self.arrival_rate

    def avg_response_time(self) -> float:
        """
        Calculate average response time (W).

        W = W_q + (1 / mu)
        """
        if self.service_rate <= 0.0:
            return float('inf')
        return self.avg_waiting_time() + (1.0 / self.service_rate)

    def throughput(self) -> float:
        """
        Calculate actual throughput.

        For stable system: throughput = lambda
        For unstable system: throughput = mu
        """
        if self.is_stable:
            return self.arrival_rate
        return self.service_rate

    def probability_n_in_system(self, n: int) -> float:
        """
        Probability of exactly n requests in system.

        P(n) = (1 - rho) * rho^n
        """
        rho = self.utilization
        if rho >= 1.0:
            return 0.0
        return (1 - rho) * (rho ** n)

    def probability_wait_exceeds(self, t: float) -> float:
        """
        Probability that waiting time exceeds t.

        P(W_q > t) = rho * exp(-mu * (1 - rho) * t)
        """
        if self.service_rate <= 0.0:
            return 0.0
        rho = self.utilization
        if rho <= 0.0:
            return 0.0
        exponent = -self.service_rate * (1 - rho) * t
        return rho * math.exp(exponent)

    def get_stats(self) -> QueueStats:
        """
        Get all queue statistics.

        Returns:
            QueueStats dataclass
        """
        return QueueStats(
            utilization=self.utilization,
            avg_queue_length=self.avg_queue_length(),
            avg_waiting_time=self.avg_waiting_time(),
            avg_response_time=self.avg_response_time(),
            throughput=self.throughput(),
            is_stable=self.is_stable
        )

    def optimal_batch_size(self, target_latency: float) -> int:
        """
        Calculate optimal batch size for target latency.

        Args:
            target_latency: Target response time in seconds

        Returns:
            Recommended batch size
        """
        if not self.is_stable:
            return 1

        W_target = target_latency
        W = self.avg_response_time()

        if W <= 0:
            return 1

        ratio = W_target / W
        batch_size = max(1, int(ratio))

        return batch_size

    def __repr__(self) -> str:
        return (f"M/M/1 Queue(lambda={self.arrival_rate:.2f}, "
                f"mu={self.service_rate:.2f}, "
                f"rho={self.utilization:.2%})")
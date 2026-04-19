"""
Unit Tests: Dynamic Batching
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
import time
from dynamicBatching import AdaptiveBatcher, M1M1Queue, Request


class TestM1M1Queue(unittest.TestCase):

    def test_initialization(self):
        """Test M/M/1 queue initialization."""
        queue = M1M1Queue(arrival_rate=10.0, service_rate=15.0)

        self.assertEqual(queue.arrival_rate, 10.0)
        self.assertEqual(queue.service_rate, 15.0)

    def test_utilization(self):
        """Test utilization calculation."""
        queue = M1M1Queue(arrival_rate=10.0, service_rate=15.0)

        self.assertAlmostEqual(queue.utilization, 10.0 / 15.0)

    def test_stability(self):
        """Test system stability check."""
        stable_queue = M1M1Queue(arrival_rate=10.0, service_rate=15.0)
        unstable_queue = M1M1Queue(arrival_rate=15.0, service_rate=10.0)

        self.assertTrue(stable_queue.is_stable)
        self.assertFalse(unstable_queue.is_stable)

    def test_avg_queue_length(self):
        """Test average queue length calculation."""
        queue = M1M1Queue(arrival_rate=10.0, service_rate=15.0)
        rho = 10.0 / 15.0
        expected = (rho ** 2) / (1 - rho)

        self.assertAlmostEqual(queue.avg_queue_length(), expected)

    def test_avg_waiting_time(self):
        """Test average waiting time calculation."""
        queue = M1M1Queue(arrival_rate=10.0, service_rate=15.0)

        Wq = queue.avg_waiting_time()

        self.assertGreater(Wq, 0)

    def test_avg_response_time(self):
        """Test average response time calculation."""
        queue = M1M1Queue(arrival_rate=10.0, service_rate=15.0)

        W = queue.avg_response_time()

        self.assertGreater(W, 0)

    def test_probability_n_in_system(self):
        """Test probability calculation."""
        queue = M1M1Queue(arrival_rate=10.0, service_rate=15.0)

        p0 = queue.probability_n_in_system(0)
        p1 = queue.probability_n_in_system(1)

        self.assertGreater(p0, 0)
        self.assertGreater(p1, 0)
        self.assertAlmostEqual(p0 + queue.probability_n_in_system(1) +
                               queue.probability_n_in_system(2), 1.0, places=1)

    def test_optimal_batch_size(self):
        """Test optimal batch size calculation."""
        queue = M1M1Queue(arrival_rate=8.0, service_rate=10.0)

        batch_size = queue.optimal_batch_size(target_latency=0.2)

        self.assertGreaterEqual(batch_size, 1)


class TestAdaptiveBatcher(unittest.TestCase):

    def setUp(self):
        self.batcher = AdaptiveBatcher(max_batch_size=8, min_batch_size=1)

    def test_initialization(self):
        """Test AdaptiveBatcher initialization."""
        self.assertEqual(self.batcher.max_batch_size, 8)
        self.assertEqual(self.batcher.min_batch_size, 1)
        self.assertEqual(len(self.batcher.request_queue), 0)

    def test_add_request(self):
        """Test adding requests."""
        request = Request(request_id=1, arrival_time=time.time(), seq_length=100)
        self.batcher.add_request(request)

        self.assertEqual(len(self.batcher.request_queue), 1)

    def test_get_batch(self):
        """Test getting a batch."""
        for i in range(5):
            request = Request(request_id=i, arrival_time=time.time(), seq_length=100)
            self.batcher.add_request(request)

        batch = self.batcher.get_batch()

        self.assertLessEqual(len(batch), self.batcher.max_batch_size)
        self.assertEqual(len(self.batcher.request_queue), 5 - len(batch))

    def test_decide_batch_size(self):
        """Test batch size decision."""
        decision = self.batcher.decide_batch_size()

        self.assertGreaterEqual(decision.batch_size, self.batcher.min_batch_size)
        self.assertLessEqual(decision.batch_size, self.batcher.max_batch_size)

    def test_complete_batch(self):
        """Test completing a batch."""
        for i in range(3):
            request = Request(request_id=i, arrival_time=time.time(), seq_length=100)
            self.batcher.add_request(request)

        batch = self.batcher.get_batch()
        self.batcher.complete_batch(batch)

        self.assertEqual(len(self.batcher.completed_requests), len(batch))


if __name__ == '__main__':
    unittest.main()
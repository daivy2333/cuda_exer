"""
Unit Tests: Memory Tracker
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
from memory import MemoryTracker, BlockAllocator


class TestMemoryTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = MemoryTracker(name="test")

    def test_initialization(self):
        """Test MemoryTracker initialization."""
        self.assertEqual(self.tracker.name, "test")
        self.assertEqual(self.tracker.allocated_bytes, 0)
        self.assertEqual(self.tracker.num_allocations, 0)

    def test_allocate(self):
        """Test allocation recording."""
        self.tracker.allocate(1024, tag="test")

        self.assertEqual(self.tracker.allocated_bytes, 1024)
        self.assertEqual(self.tracker.num_allocations, 1)

    def test_free(self):
        """Test deallocation recording."""
        self.tracker.allocate(1024, tag="test")
        self.tracker.free(1024, tag="test")

        self.assertEqual(self.tracker.num_frees, 1)

    def test_peak_tracking(self):
        """Test peak allocation tracking."""
        self.tracker.allocate(1000)
        self.assertEqual(self.tracker.peak_bytes, 1000)

        self.tracker.allocate(2000)
        self.assertEqual(self.tracker.peak_bytes, 2000)

        self.tracker.free(1000)
        self.assertEqual(self.tracker.peak_bytes, 2000)

    def test_get_current_stats(self):
        """Test getting current statistics."""
        self.tracker.allocate(1024)

        stats = self.tracker.get_current_stats()

        self.assertEqual(stats.allocated_bytes, 1024)
        self.assertEqual(stats.num_allocations, 1)
        self.assertGreater(stats.timestamp, 0)


class TestBlockAllocator(unittest.TestCase):

    def setUp(self):
        self.alloc = BlockAllocator(num_blocks=20, block_size=4096)

    def test_initialization(self):
        """Test BlockAllocator initialization."""
        self.assertEqual(self.alloc.num_blocks, 20)
        self.assertEqual(self.alloc.block_size, 4096)
        self.assertEqual(self.alloc.get_num_free(), 20)

    def test_allocate_single(self):
        """Test allocating single block."""
        blocks = self.alloc.allocate(1)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(self.alloc.get_num_free(), 19)
        self.assertEqual(self.alloc.get_num_used(), 1)

    def test_allocate_multiple(self):
        """Test allocating multiple blocks."""
        blocks = self.alloc.allocate(5)

        self.assertEqual(len(blocks), 5)
        self.assertEqual(self.alloc.get_num_free(), 15)

    def test_free(self):
        """Test freeing blocks."""
        blocks = self.alloc.allocate(5)
        self.alloc.free(blocks)

        self.assertEqual(self.alloc.get_num_free(), 20)

    def test_free_all(self):
        """Test freeing all blocks."""
        self.alloc.allocate(5)
        self.alloc.free_all()

        self.assertEqual(self.alloc.get_num_free(), 20)

    def test_can_allocate(self):
        """Test allocation check."""
        self.assertTrue(self.alloc.can_allocate(20))
        self.assertTrue(self.alloc.can_allocate(10))
        self.assertFalse(self.alloc.can_allocate(21))

    def test_utilization(self):
        """Test utilization calculation."""
        self.alloc.allocate(5)

        self.assertAlmostEqual(self.alloc.get_utilization(), 0.25)


if __name__ == '__main__':
    unittest.main()
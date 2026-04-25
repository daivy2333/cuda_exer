"""
Unit Tests: Block Table
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
from pagedAttention import BlockTable, Block


class TestBlockTable(unittest.TestCase):

    def setUp(self):
        self.bt = BlockTable(block_size=16, num_blocks=10)

    def test_initialization(self):
        """Test BlockTable initialization."""
        self.assertEqual(self.bt.block_size, 16)
        self.assertEqual(self.bt.num_blocks, 10)
        self.assertEqual(self.bt.get_num_free_blocks(), 10)
        self.assertEqual(self.bt.get_num_used_blocks(), 0)

    def test_allocate_single_sequence(self):
        """Test allocating blocks for a single sequence."""
        block_ids = self.bt.allocate(seq_id=1, num_tokens=50)

        self.assertEqual(len(block_ids), 4)
        self.assertEqual(self.bt.get_num_free_blocks(), 6)
        self.assertEqual(self.bt.get_num_used_blocks(), 4)
        self.assertEqual(self.bt.mappings[1], block_ids)

    def test_allocate_multiple_sequences(self):
        """Test allocating blocks for multiple sequences."""
        blocks1 = self.bt.allocate(seq_id=1, num_tokens=30)
        blocks2 = self.bt.allocate(seq_id=2, num_tokens=20)

        self.assertEqual(len(blocks1), 2)
        self.assertEqual(len(blocks2), 2)
        self.assertEqual(self.bt.get_num_used_blocks(), 4)

    def test_free_sequence(self):
        """Test freeing a sequence's blocks."""
        self.bt.allocate(seq_id=1, num_tokens=50)
        self.bt.free(seq_id=1)

        self.assertEqual(self.bt.get_num_free_blocks(), 10)
        self.assertNotIn(1, self.bt.mappings)

    def test_free_and_reallocate(self):
        """Test that freed blocks can be reused."""
        initial_free = self.bt.get_num_free_blocks()
        blocks1 = self.bt.allocate(seq_id=1, num_tokens=50)
        free_before_free = self.bt.get_num_free_blocks()

        self.bt.free(seq_id=1)
        free_after = self.bt.get_num_free_blocks()

        self.assertEqual(free_after - free_before_free, len(blocks1))
        self.assertEqual(free_after, initial_free)

        blocks2 = self.bt.allocate(seq_id=2, num_tokens=30)
        self.assertEqual(len(blocks2), 2)
        self.assertEqual(self.bt.get_num_free_blocks(), initial_free - 2)

    def test_allocation_failure(self):
        """Test allocation fails when out of blocks."""
        for i in range(10):
            self.bt.allocate(seq_id=i, num_tokens=16)

        with self.assertRaises(RuntimeError):
            self.bt.allocate(seq_id=99, num_tokens=16)

    def test_fragmentation_rate(self):
        """Test fragmentation rate calculation."""
        block_ids = self.bt.allocate(seq_id=1, num_tokens=50)

        frag = self.bt.get_fragmentation_rate(seq_id=1)

        self.assertGreaterEqual(frag, 0.0)
        self.assertLessEqual(frag, 1.0)

    def test_utilization(self):
        """Test overall utilization calculation."""
        self.bt.allocate(seq_id=1, num_tokens=50)

        util = self.bt.get_utilization()

        self.assertGreaterEqual(util, 0.0)
        self.assertLessEqual(util, 1.0)

    def test_reset(self):
        """Test resetting the block table."""
        self.bt.allocate(seq_id=1, num_tokens=50)
        self.bt.allocate(seq_id=2, num_tokens=30)

        self.bt.reset()

        self.assertEqual(self.bt.get_num_free_blocks(), 10)
        self.assertEqual(len(self.bt.mappings), 0)


class TestBlock(unittest.TestCase):

    def test_block_is_full(self):
        """Test block fullness check."""
        block = Block(block_id=0, size=16)

        self.assertFalse(block.is_full())

        block.tokens = [0] * 16

        self.assertTrue(block.is_full())

    def test_block_remaining_capacity(self):
        """Test remaining capacity calculation."""
        block = Block(block_id=0, size=16)

        self.assertEqual(block.remaining_capacity(), 16)

        block.tokens = [0] * 10

        self.assertEqual(block.remaining_capacity(), 6)


if __name__ == '__main__':
    unittest.main()
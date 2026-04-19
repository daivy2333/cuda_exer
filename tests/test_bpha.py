"""
Unit Tests: BPHA
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
import torch
import numpy as np
from bpha import BPHAOperator, bpha_forward, bpha_backward


class TestBPHAOperator(unittest.TestCase):

    def setUp(self):
        self.hidden_dim = 64
        self.block_size = 16
        self.op = BPHAOperator(hidden_dim=self.hidden_dim, block_size=self.block_size)

    def test_initialization(self):
        """Test BPHAOperator initialization."""
        self.assertEqual(self.op.hidden_dim, 64)
        self.assertEqual(self.op.block_size, 16)
        self.assertEqual(self.op.head_dim, 64)

    def test_forward_single_block(self):
        """Test forward pass with single block."""
        batch_size, q_len = 1, 1
        query = torch.randn(batch_size, q_len, self.hidden_dim)
        k_block = torch.randn(batch_size, self.block_size, self.hidden_dim)
        v_block = torch.randn(batch_size, self.block_size, self.hidden_dim)

        output = self.op.forward(query, [(k_block, v_block)], [0])

        self.assertEqual(output.shape, (batch_size, q_len, self.hidden_dim))

    def test_forward_multiple_blocks(self):
        """Test forward pass with multiple blocks."""
        batch_size, q_len = 2, 1
        seq_len = 32
        num_blocks = seq_len // self.block_size

        query = torch.randn(batch_size, q_len, self.hidden_dim)

        kv_blocks = []
        block_offsets = []
        for i in range(num_blocks):
            k = torch.randn(batch_size, self.block_size, self.hidden_dim)
            v = torch.randn(batch_size, self.block_size, self.hidden_dim)
            kv_blocks.append((k, v))
            block_offsets.append(i * self.block_size)

        output = self.op.forward(query, kv_blocks, block_offsets)

        self.assertEqual(output.shape, (batch_size, q_len, self.hidden_dim))

    def test_output_numerical_stability(self):
        """Test that output values are reasonable."""
        query = torch.randn(1, 1, self.hidden_dim)
        k = torch.randn(1, self.block_size, self.hidden_dim)
        v = torch.randn(1, self.block_size, self.hidden_dim)

        output, _ = self.op.forward(query, [(k, v)], [0])

        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())


class TestBPHAForward(unittest.TestCase):

    def test_bpha_forward_basic(self):
        """Test basic bpha_forward function."""
        query = torch.randn(1, 1, 64)
        k = torch.randn(1, 16, 64)
        v = torch.randn(1, 16, 64)

        output = bpha_forward(query, [(k, v)], [0])

        self.assertEqual(output.shape, query.shape)

    def test_bpha_forward_multiple_blocks(self):
        """Test bpha_forward with multiple blocks."""
        query = torch.randn(2, 1, 64)

        kv_blocks = []
        offsets = []
        for i in range(3):
            k = torch.randn(2, 16, 64)
            v = torch.randn(2, 16, 64)
            kv_blocks.append((k, v))
            offsets.append(i * 16)

        output = bpha_forward(query, kv_blocks, offsets)

        self.assertEqual(output.shape, (2, 1, 64))


class TestBPHAComparison(unittest.TestCase):

    def test_bpha_matches_standard(self):
        """Test that BPHA matches standard attention."""
        batch_size, q_len, seq_len, hidden_dim = 1, 1, 32, 64
        block_size = 16

        query = torch.randn(batch_size, q_len, hidden_dim)
        keys = torch.randn(batch_size, seq_len, hidden_dim)
        values = torch.randn(batch_size, seq_len, hidden_dim)

        num_blocks = seq_len // block_size
        kv_blocks = []
        offsets = []
        for i in range(num_blocks):
            start = i * block_size
            k = keys[:, start:start+block_size, :]
            v = values[:, start:start+block_size, :]
            kv_blocks.append((k, v))
            offsets.append(start)

        scale = 1.0 / (hidden_dim ** 0.5)
        scores = torch.matmul(query, keys.transpose(-2, -1)) * scale
        expected = torch.matmul(torch.softmax(scores, dim=-1), values)

        bpha_op = BPHAOperator(hidden_dim=hidden_dim, block_size=block_size)
        result = bpha_op.forward(query, kv_blocks, offsets)

        max_diff = torch.max(torch.abs(expected - result)).item()

        self.assertLess(max_diff, 1e-4)


if __name__ == '__main__':
    unittest.main()
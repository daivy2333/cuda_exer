import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestModelLoader(unittest.TestCase):
    def test_load_qwen_model(self):
        """Test loading Qwen model from local path."""
        from qwen_adapter.model_loader import load_qwen_model

        model_path = os.path.join(
            os.path.dirname(__file__), '..', 'model',
            'models--Qwen--Qwen2.5-3B-Instruct', 'snapshots',
            'aa8e72537993ba99e69dfaafa59ed015b17504d1'
        )

        model, tokenizer = load_qwen_model(model_path)

        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)
        self.assertEqual(model.config.hidden_size, 2048)
        self.assertEqual(model.config.num_attention_heads, 16)


class TestBPHAAttention(unittest.TestCase):
    def setUp(self):
        self.hidden_size = 2048
        self.num_heads = 16
        self.num_kv_heads = 2
        self.head_dim = 128
        self.block_size = 16

    def test_bpha_attention_init(self):
        """Test BPHAAttention initialization with GQA config."""
        from qwen_adapter.bpha_attention import BPHAAttention

        attn = BPHAAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            block_size=self.block_size,
        )

        self.assertEqual(attn.hidden_size, 2048)
        self.assertEqual(attn.num_heads, 16)
        self.assertEqual(attn.num_kv_heads, 2)
        self.assertEqual(attn.head_dim, 128)
        self.assertEqual(attn.num_groups, 8)  # 16 / 2 = 8

    def test_bpha_attention_forward(self):
        """Test BPHAAttention forward pass."""
        import torch
        from qwen_adapter.bpha_attention import BPHAAttention

        attn = BPHAAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            block_size=self.block_size,
        )

        batch_size = 1
        seq_len = 32
        hidden_states = torch.randn(batch_size, seq_len, self.hidden_size)

        # Create mock KV cache
        kv_blocks = []
        for i in range(seq_len // self.block_size):
            k = torch.randn(batch_size, self.num_kv_heads, self.block_size, self.head_dim)
            v = torch.randn(batch_size, self.num_kv_heads, self.block_size, self.head_dim)
            kv_blocks.append((k, v))

        output = attn.forward(hidden_states, kv_blocks)

        self.assertEqual(output.shape, (batch_size, seq_len, self.hidden_size))
        self.assertFalse(torch.isnan(output).any())


if __name__ == '__main__':
    unittest.main()
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


class TestKVCacheManager(unittest.TestCase):
    def setUp(self):
        self.num_layers = 36
        self.num_kv_heads = 2
        self.head_dim = 128
        self.block_size = 16
        self.max_blocks = 100

    def test_kv_cache_manager_init(self):
        """Test KVCacheManager initialization."""
        from qwen_adapter.kv_cache_manager import KVCacheManager

        manager = KVCacheManager(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            block_size=self.block_size,
            max_blocks=self.max_blocks,
        )

        self.assertEqual(manager.num_layers, 36)
        self.assertEqual(manager.num_kv_heads, 2)
        self.assertEqual(manager.block_size, 16)

    def test_allocate_and_store(self):
        """Test allocating and storing KV blocks."""
        import torch
        from qwen_adapter.kv_cache_manager import KVCacheManager

        manager = KVCacheManager(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            block_size=self.block_size,
            max_blocks=self.max_blocks,
        )

        # Simulate new tokens
        k_new = torch.randn(1, self.num_kv_heads, 10, self.head_dim)
        v_new = torch.randn(1, self.num_kv_heads, 10, self.head_dim)

        manager.allocate_sequence(seq_id=1, num_tokens=10)
        manager.store_kv(layer_idx=0, seq_id=1, k_new=k_new, v_new=v_new)

        blocks = manager.get_kv_blocks(layer_idx=0, seq_id=1)
        self.assertTrue(len(blocks) > 0)

    def test_memory_stats(self):
        """Test memory statistics reporting."""
        from qwen_adapter.kv_cache_manager import KVCacheManager

        manager = KVCacheManager(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            block_size=self.block_size,
            max_blocks=self.max_blocks,
        )

        stats = manager.get_memory_stats()

        self.assertIn("total_blocks", stats)
        self.assertIn("used_blocks", stats)
        self.assertIn("memory_bytes", stats)


class TestReplaceAttention(unittest.TestCase):
    def test_replace_attention_in_model(self):
        """Test replacing attention layers in model."""
        from qwen_adapter.model_loader import load_qwen_model
        from qwen_adapter.replace_attention import replace_attention_with_bpha
        from qwen_adapter.kv_cache_manager import KVCacheManager

        model_path = os.path.join(
            os.path.dirname(__file__), '..', 'model',
            'models--Qwen--Qwen2.5-3B-Instruct', 'snapshots',
            'aa8e72537993ba99e69dfaafa59ed015b17504d1'
        )

        model, tokenizer = load_qwen_model(model_path, device="cpu")

        # Count original attention layers
        original_attn_count = 0
        for name, module in model.named_modules():
            if 'self_attn' in name and hasattr(module, 'q_proj'):
                original_attn_count += 1

        self.assertTrue(original_attn_count > 0)

        # Replace attention
        kv_manager = KVCacheManager(
            num_layers=model.config.num_hidden_layers,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=model.config.hidden_size // model.config.num_attention_heads,
            block_size=16,
            max_blocks=100,
        )

        replace_attention_with_bpha(model, kv_manager)

        # Verify replacement - check for kv_manager attribute
        bpha_count = 0
        for name, module in model.named_modules():
            if hasattr(module, 'forward') and hasattr(module, 'kv_manager'):
                bpha_count += 1

        self.assertEqual(bpha_count, original_attn_count)


if __name__ == '__main__':
    unittest.main()
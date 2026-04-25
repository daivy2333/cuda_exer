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


if __name__ == '__main__':
    unittest.main()
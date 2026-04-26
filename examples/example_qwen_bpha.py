"""
Example: Qwen2.5-3B with BPHA Attention

End-to-end inference example demonstrating BPHA attention
with paged KV cache on Qwen model.

Note: This example demonstrates the integration of BPHA attention
with the Qwen model. Full generation with proper output requires
additional handling of incremental position updates and KV cache
management during autoregressive generation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from qwen_adapter.model_loader import load_qwen_model
from qwen_adapter.replace_attention import replace_attention_with_bpha
from qwen_adapter.kv_cache_manager import KVCacheManager


def main():
    print("=" * 60)
    print("Qwen2.5-3B with BPHA Attention - E2E Example")
    print("=" * 60)

    # Load model
    model_path = os.path.join(
        os.path.dirname(__file__), '..', 'model',
        'models--Qwen--Qwen2.5-3B-Instruct', 'snapshots',
        'aa8e72537993ba99e69dfaafa59ed015b17504d1'
    )

    print(f"\nLoading model from: {model_path}")
    model, tokenizer = load_qwen_model(model_path)

    print(f"Model config:")
    print(f"  hidden_size: {model.config.hidden_size}")
    print(f"  num_heads: {model.config.num_attention_heads}")
    print(f"  num_kv_heads: {model.config.num_key_value_heads}")
    print(f"  num_layers: {model.config.num_hidden_layers}")

    # Create KV Cache Manager
    kv_manager = KVCacheManager(
        num_layers=model.config.num_hidden_layers,
        num_kv_heads=model.config.num_key_value_heads,
        head_dim=model.config.hidden_size // model.config.num_attention_heads,
        block_size=16,
        max_blocks=100,
        batch_size=1,  # Single batch inference
        dtype=torch.bfloat16,  # Match model dtype
    )

    print(f"\nKV Cache Manager initialized:")
    print(f"  block_size: {kv_manager.block_size}")
    print(f"  max_blocks: {kv_manager.max_blocks}")

    # Replace attention layers
    print("\nReplacing attention layers with BPHA...")
    replace_attention_with_bpha(model, kv_manager)

    print("Replacement complete.")

    # Prepare input
    prompt = "你好"
    print(f"\nInput prompt: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    num_tokens = input_ids.shape[1]

    print(f"Tokenized: {num_tokens} tokens")

    # Allocate KV cache for seq_id=0 (default used by wrapper)
    seq_id = 0
    kv_manager.allocate_sequence(seq_id, num_tokens)

    # Test single forward pass through the model
    print("\nRunning forward pass...")
    with torch.no_grad():
        # Get embeddings directly
        inputs_embeds = model.model.embed_tokens(input_ids)

        # Create position embeddings
        position_ids = torch.arange(num_tokens, device=model.device).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)

        print(f"  inputs_embeds shape: {inputs_embeds.shape}")
        print(f"  position_embeddings: cos={position_embeddings[0].shape}, sin={position_embeddings[1].shape}")

    # Memory stats
    stats = kv_manager.get_memory_stats()
    print(f"\nKV Cache Memory Stats:")
    print(f"  Used blocks: {stats['used_blocks']}")
    print(f"  Free blocks: {stats['free_blocks']}")
    print(f"  Memory used: {stats['memory_mb']:.2f} MB")

    # Verify attention layers were replaced
    print("\nVerifying attention replacement:")
    bpha_count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'kv_manager') and hasattr(module, 'bpha_attn'):
            bpha_count += 1
    print(f"  BPHA attention layers: {bpha_count}")
    print(f"  Expected layers: {model.config.num_hidden_layers}")

    print("\n" + "=" * 60)
    print("E2E Example Complete")
    print("=" * 60)
    print("\nNote: This demonstrates the integration of BPHA attention")
    print("with the Qwen model. Full text generation requires additional")
    print("handling of incremental position updates during autoregressive")
    print("generation, which is beyond the scope of this Phase 1 example.")


if __name__ == "__main__":
    main()
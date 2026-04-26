"""
Example: Qwen2.5-3B with BPHA Attention

End-to-end inference example demonstrating BPHA attention
with paged KV cache on Qwen model for text generation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from qwen_adapter.model_loader import load_qwen_model
from qwen_adapter.replace_attention import replace_attention_with_bpha
from qwen_adapter.kv_cache_manager import KVCacheManager


def forward_with_position(model, input_ids, position_ids, start_pos=0):
    """Run forward pass with explicit start position for KV cache.

    This function passes the start_pos to the attention layers so they
    know where to store new KV vectors during incremental generation.
    """
    # We need to modify the model's forward to pass start_pos through kwargs
    # The simplest approach is to use the model's forward directly
    # and rely on the wrapper to handle the position tracking

    # For now, we'll use a hack: temporarily modify the wrapper's forward
    # to pass start_pos as a global variable
    from qwen_adapter.replace_attention import BPHAAttentionWrapper

    # Store the original forward
    original_forward = BPHAAttentionWrapper.forward

    # Create a wrapper that passes start_pos
    def forward_with_start_pos(self, hidden_states, position_embeddings, attention_mask=None, past_key_value=None, cache_position=None, **kwargs):
        kwargs['_bpha_start_pos'] = start_pos
        return original_forward(self, hidden_states, position_embeddings, attention_mask, past_key_value, cache_position, **kwargs)

    # Temporarily replace the forward
    BPHAAttentionWrapper.forward = forward_with_start_pos

    try:
        outputs = model.forward(input_ids, position_ids=position_ids)
    finally:
        # Restore the original forward
        BPHAAttentionWrapper.forward = original_forward

    return outputs


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

    # Create KV Cache Manager with enough blocks for generation
    block_size = 16
    max_blocks = 500  # Enough for ~8000 tokens
    kv_manager = KVCacheManager(
        num_layers=model.config.num_hidden_layers,
        num_kv_heads=model.config.num_key_value_heads,
        head_dim=model.config.hidden_size // model.config.num_attention_heads,
        block_size=block_size,
        max_blocks=max_blocks,
        batch_size=1,
        dtype=torch.bfloat16,
    )

    print(f"\nKV Cache Manager initialized:")
    print(f"  block_size: {kv_manager.block_size}")
    print(f"  max_blocks: {kv_manager.max_blocks}")
    print(f"  max_tokens: {max_blocks * block_size}")

    # Replace attention layers
    print("\nReplacing attention layers with BPHA...")
    replace_attention_with_bpha(model, kv_manager)

    print("Replacement complete.")

    # Prepare input - use full prompt as requested
    prompt = "你好，请介绍一下你自己。"
    print(f"\nInput prompt: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    num_tokens = input_ids.shape[1]

    print(f"Tokenized: {num_tokens} tokens")
    print(f"Token IDs: {input_ids[0].tolist()}")

    # Text generation parameters
    max_new_tokens = 100
    seq_id = 0

    # Allocate KV cache for initial tokens + expected new tokens
    total_expected = num_tokens + max_new_tokens
    kv_manager.allocate_sequence(seq_id, total_expected)

    print(f"\nStarting text generation...")
    print(f"  max_new_tokens: {max_new_tokens}")
    print("-" * 60)

    # Generation loop
    generated_ids = input_ids.clone()
    current_pos = num_tokens

    with torch.no_grad():
        # First forward pass: process the prompt (full sequence)
        outputs = forward_with_position(
            model,
            generated_ids,
            torch.arange(num_tokens, device=model.device).unsqueeze(0),
            start_pos=0
        )

        # Get logits for next token
        logits = outputs.logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)

        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            print("[EOS token reached at step 1]")
        else:
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            current_pos += 1

            # Continue generation - pass only the last token with correct position
            for step in range(1, max_new_tokens):
                # Forward pass with single token at correct position
                outputs = forward_with_position(
                    model,
                    next_token,
                    torch.tensor([[current_pos - 1]], device=model.device),
                    start_pos=current_pos - 1
                )

                # Get logits
                logits = outputs.logits[:, -1, :]
                next_token = logits.argmax(dim=-1, keepdim=True)

                # Check for EOS
                if next_token.item() == tokenizer.eos_token_id:
                    print(f"\n[EOS token reached at step {step + 1}]")
                    break

                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                current_pos += 1

                # Print progress every 10 tokens
                if (step + 1) % 10 == 0:
                    new_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
                    print(f"Step {step + 1}: +'{new_text}'")

    # Final output
    print("-" * 60)
    print("\nFinal generated text:")
    print("=" * 60)
    final_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(final_text)
    print("=" * 60)

    # Memory stats
    stats = kv_manager.get_memory_stats()
    print(f"\nKV Cache Memory Stats:")
    print(f"  Used blocks: {stats['used_blocks']}")
    print(f"  Free blocks: {stats['free_blocks']}")
    print(f"  Memory used: {stats['memory_mb']:.2f} MB")
    print(f"  Tokens generated: {generated_ids.shape[1] - num_tokens}")

    # Verify attention layers were replaced
    print("\nVerifying attention replacement:")
    bpha_count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'kv_manager') and hasattr(module, 'bpha_attn'):
            bpha_count += 1
    print(f"  BPHA attention layers: {bpha_count}")
    print(f"  Expected layers: {model.config.num_hidden_layers}")

    print("\n" + "=" * 60)
    print("E2E Text Generation Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
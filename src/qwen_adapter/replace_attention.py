"""
Attention Replacement Utility: Replace Qwen's native attention with BPHA.

This module provides functionality to replace Qwen2.5's native attention layers
with BPHA (Block-Paged Hybrid Attention) wrappers that integrate with the
KV cache manager for paged attention.
"""

from typing import Optional
import torch
import torch.nn as nn

from qwen_adapter.bpha_attention import BPHAAttention
from qwen_adapter.kv_cache_manager import KVCacheManager


class BPHAAttentionWrapper(nn.Module):
    """Wraps original attention layer with BPHA and KV cache integration.

    This wrapper:
    - Preserves the original attention's projection weights
    - Uses BPHA for attention computation
    - Integrates with KVCacheManager for paged KV cache

    Args:
        original_attn: Original Qwen2 attention layer
        bpha_attn: BPHA attention module
        kv_manager: KV cache manager instance
        layer_idx: Index of this layer in the transformer
    """

    def __init__(
        self,
        original_attn: nn.Module,
        bpha_attn: BPHAAttention,
        kv_manager: KVCacheManager,
        layer_idx: int,
    ):
        super().__init__()

        self.original_attn = original_attn
        self.bpha_attn = bpha_attn
        self.kv_manager = kv_manager
        self.layer_idx = layer_idx

        # Store config for reference
        self.hidden_size = bpha_attn.hidden_size
        self.num_heads = bpha_attn.num_heads
        self.num_kv_heads = bpha_attn.num_kv_heads
        self.head_dim = bpha_attn.head_dim

        # Copy weights from original attention to BPHA
        self._copy_weights()

    def _copy_weights(self) -> None:
        """Copy projection weights from original attention to BPHA."""
        # Map Qwen attention projections to BPHA projections
        # Qwen uses: q_proj, k_proj, v_proj, o_proj
        # BPHA uses: q_proj, k_proj, v_proj, o_proj

        if hasattr(self.original_attn, 'q_proj'):
            self.bpha_attn.q_proj.weight.data = self.original_attn.q_proj.weight.data.clone()

        if hasattr(self.original_attn, 'k_proj'):
            self.bpha_attn.k_proj.weight.data = self.original_attn.k_proj.weight.data.clone()

        if hasattr(self.original_attn, 'v_proj'):
            self.bpha_attn.v_proj.weight.data = self.original_attn.v_proj.weight.data.clone()

        if hasattr(self.original_attn, 'o_proj'):
            self.bpha_attn.o_proj.weight.data = self.original_attn.o_proj.weight.data.clone()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple:
        """Forward pass using BPHA with KV cache integration.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Attention mask (unused in BPHA, kept for compatibility)
            position_ids: Position IDs (unused, kept for compatibility)
            past_key_value: Past KV cache (unused, we use KVCacheManager)
            output_attentions: Whether to output attention weights
            use_cache: Whether to use KV cache
            **kwargs: Additional arguments (may contain seq_id)

        Returns:
            Tuple of (output, None, None) for compatibility with Qwen interface
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Get sequence ID from kwargs (for KV cache management)
        seq_id = kwargs.get('seq_id', 0)

        # Project Q, K, V using BPHA projections
        q = self.bpha_attn.q_proj(hidden_states)
        k = self.bpha_attn.k_proj(hidden_states)
        v = self.bpha_attn.v_proj(hidden_states)

        # Reshape for KV cache
        k_cache, v_cache = self.bpha_attn.reshape_for_cache(k, v)

        # Get existing KV blocks from manager
        kv_blocks = []
        try:
            kv_blocks = self.kv_manager.get_kv_blocks(self.layer_idx, seq_id)
        except RuntimeError:
            # Sequence not allocated yet, start with empty KV
            pass

        # Store new K, V in cache
        try:
            # Try to store - if sequence not allocated, allocate first
            self.kv_manager.store_kv(self.layer_idx, seq_id, k_cache, v_cache)
        except RuntimeError:
            # Sequence not allocated, allocate and store
            self.kv_manager.allocate_sequence(seq_id, seq_len)
            self.kv_manager.store_kv(self.layer_idx, seq_id, k_cache, v_cache)
            # Get KV blocks again after allocation
            kv_blocks = self.kv_manager.get_kv_blocks(self.layer_idx, seq_id)

        # Compute attention using BPHA
        output = self.bpha_attn(hidden_states, kv_blocks)

        # Return in Qwen's expected format: (hidden_states, None, past_key_value)
        return (output, None, None)


def replace_attention_with_bpha(
    model: nn.Module,
    kv_manager: KVCacheManager,
) -> None:
    """Replace all attention layers in model with BPHA wrappers.

    This function iterates over the model's transformer layers and replaces
    each self_attn module with a BPHAAttentionWrapper.

    Args:
        model: Qwen2ForCausalLM model
        kv_manager: KV cache manager for paged attention

    Note:
        This modifies the model in-place. The original attention layers
        are wrapped, not replaced entirely, to preserve weights.
    """
    # Get model config
    config = model.config

    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    num_layers = config.num_hidden_layers

    # Access transformer layers
    # Qwen2 structure: model.model.layers[i].self_attn
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        raise ValueError("Model does not have expected structure (model.model.layers)")

    for layer_idx, layer in enumerate(layers):
        if not hasattr(layer, 'self_attn'):
            continue

        original_attn = layer.self_attn

        # Create BPHA attention for this layer
        bpha_attn = BPHAAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            block_size=16,  # Default block size
            layer_idx=layer_idx,
        )

        # Create wrapper
        wrapper = BPHAAttentionWrapper(
            original_attn=original_attn,
            bpha_attn=bpha_attn,
            kv_manager=kv_manager,
            layer_idx=layer_idx,
        )

        # Replace self_attn with wrapper
        layer.self_attn = wrapper


def get_attention_info(model: nn.Module) -> dict:
    """Get information about attention layers in the model.

    Args:
        model: Qwen2ForCausalLM model

    Returns:
        Dict with attention layer information:
        - num_layers: Number of transformer layers
        - hidden_size: Hidden dimension
        - num_heads: Number of attention heads
        - num_kv_heads: Number of KV heads (for GQA)
    """
    config = model.config

    return {
        'num_layers': config.num_hidden_layers,
        'hidden_size': config.hidden_size,
        'num_heads': config.num_attention_heads,
        'num_kv_heads': config.num_key_value_heads,
        'head_dim': config.hidden_size // config.num_attention_heads,
    }
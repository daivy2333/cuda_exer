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

        # Copy weights from original attention to BPHA
        self._copy_weights()

    def _copy_weights(self) -> None:
        """Copy projection weights and biases from original attention to BPHA."""
        # Map Qwen attention projections to BPHA projections
        # Qwen uses: q_proj, k_proj, v_proj, o_proj
        # BPHA uses: q_proj, k_proj, v_proj, o_proj

        if hasattr(self.original_attn, 'q_proj'):
            self.bpha_attn.q_proj.weight.data = self.original_attn.q_proj.weight.data.clone()
            if self.original_attn.q_proj.bias is not None:
                self.bpha_attn.q_proj.bias.data = self.original_attn.q_proj.bias.data.clone()

        if hasattr(self.original_attn, 'k_proj'):
            self.bpha_attn.k_proj.weight.data = self.original_attn.k_proj.weight.data.clone()
            if self.original_attn.k_proj.bias is not None:
                self.bpha_attn.k_proj.bias.data = self.original_attn.k_proj.bias.data.clone()

        if hasattr(self.original_attn, 'v_proj'):
            self.bpha_attn.v_proj.weight.data = self.original_attn.v_proj.weight.data.clone()
            if self.original_attn.v_proj.bias is not None:
                self.bpha_attn.v_proj.bias.data = self.original_attn.v_proj.bias.data.clone()

        if hasattr(self.original_attn, 'o_proj'):
            self.bpha_attn.o_proj.weight.data = self.original_attn.o_proj.weight.data.clone()
            # o_proj doesn't have bias in Qwen2

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple:
        """Forward pass using BPHA with KV cache integration.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            position_embeddings: Tuple of (cos, sin) tensors for rotary embeddings
            attention_mask: Attention mask (unused in BPHA, kept for compatibility)
            past_key_value: Past KV cache (unused, we use KVCacheManager)
            cache_position: Cache position (unused, kept for compatibility)
            **kwargs: Additional arguments (may contain seq_id)

        Returns:
            Tuple of (output, None) for compatibility with Qwen2.5 interface
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Get sequence ID from kwargs (for KV cache management)
        seq_id = kwargs.get('seq_id', 0)  # Default 0 means single-request mode

        # Get rotary embeddings (cos, sin)
        cos, sin = position_embeddings

        # Project Q, K, V using BPHA projections
        q = self.bpha_attn.q_proj(hidden_states)
        k = self.bpha_attn.k_proj(hidden_states)
        v = self.bpha_attn.v_proj(hidden_states)

        # Reshape and transpose to match Qwen2's expected shape: [batch, num_heads, seq_len, head_dim]
        batch_size, seq_len, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_len, self.bpha_attn.num_heads, self.bpha_attn.head_dim)
        q = q.view(hidden_shape).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        k = k.view(batch_size, seq_len, self.bpha_attn.num_kv_heads, self.bpha_attn.head_dim).transpose(1, 2)  # [batch, num_kv_heads, seq_len, head_dim]
        v = v.view(batch_size, seq_len, self.bpha_attn.num_kv_heads, self.bpha_attn.head_dim).transpose(1, 2)  # [batch, num_kv_heads, seq_len, head_dim]

        # Apply rotary embeddings to Q and K (using Qwen2's formula)
        # cos, sin are [batch, seq_len, head_dim]
        q_rot, k_rot = self._apply_rotary_pos_emb(q, k, cos, sin)

        # Reshape for KV cache: [batch, num_kv_heads, seq_len, head_dim] (already in this shape)
        k_cache = k_rot
        v_cache = v

        # Check if sequence is allocated
        try:
            self.kv_manager.get_kv_blocks(self.layer_idx, seq_id)
            # Sequence exists, sequence is already allocated
            pass
        except RuntimeError:
            # Sequence not allocated, allocate it
            self.kv_manager.allocate_sequence(seq_id, seq_len)

        # Get the starting position for this forward pass
        start_pos = kwargs.get('_bpha_start_pos', 0)

        # Determine if this is incremental generation (single token) or prefill
        if seq_len == 1 and start_pos > 0:
            # Incremental generation: store at the specified position
            # Check if THIS layer has already stored this token
            layer_block_ids = self.kv_manager.layer_block_tables[self.layer_idx].get(seq_id, [])
            if layer_block_ids:
                # Calculate total tokens stored in THIS layer's blocks
                layer_stored = 0
                for block_id in layer_block_ids:
                    key = (self.layer_idx, block_id)
                    if key in self.kv_manager.physical_blocks:
                        layer_stored += self.kv_manager.physical_blocks[key].num_tokens

                already_stored_this_layer = layer_stored > start_pos
                if not already_stored_this_layer:
                    # This layer hasn't stored this token yet
                    self.kv_manager.store_kv(self.layer_idx, seq_id, k_cache, v_cache)
        else:
            # Prefill: store all tokens at positions 0 to seq_len-1
            self.kv_manager.store_kv(self.layer_idx, seq_id, k_cache, v_cache)

        # Get all KV blocks (including newly stored)
        kv_blocks = self.kv_manager.get_kv_blocks(self.layer_idx, seq_id)

        # Q is already in correct shape: [batch, num_heads, seq_len, head_dim]
        q_attn = q_rot

        # Compute attention using BPHA's attention computation
        output = self._compute_attention(q_attn, kv_blocks, hidden_states)

        # Return in Qwen2.5's expected format: (hidden_states, None)
        return (output, None)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input (Qwen2's rotate_half)."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1,
    ) -> tuple:
        """Apply Rotary Position Embedding to query and key tensors (Qwen2's formula).

        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
            cos: Cosine embeddings [batch, seq_len, head_dim]
            sin: Sine embeddings [batch, seq_len, head_dim]
            unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting

        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def _apply_rotary_emb(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary embeddings to tensor.

        Args:
            x: Tensor [batch, seq_len, num_heads, head_dim]
            cos: Cosine embeddings [batch, seq_len, head_dim]
            sin: Sine embeddings [batch, seq_len, head_dim]

        Returns:
            Rotated tensor [batch, seq_len, num_heads, head_dim]
        """
        # Unsqueeze cos/sin to match x's dimensions
        # cos/sin: [batch, seq_len, head_dim] -> [batch, seq_len, 1, head_dim]
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)

        # Rotary embedding applies to pairs of dimensions
        # Split head_dim into two halves
        head_dim = x.shape[-1]
        x1 = x[..., :head_dim // 2]
        x2 = x[..., head_dim // 2:]

        # Ensure cos/sin match the half dimension
        cos_half = cos[..., :head_dim // 2]
        sin_half = sin[..., :head_dim // 2]

        # Apply rotation
        rotated_x1 = x1 * cos_half - x2 * sin_half
        rotated_x2 = x1 * sin_half + x2 * cos_half

        # Concatenate back
        return torch.cat([rotated_x1, rotated_x2], dim=-1)

    def _apply_rotary_emb_gqa(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary embeddings for GQA (fewer KV heads).

        Args:
            x: Tensor [batch, seq_len, num_kv_heads, head_dim]
            cos: Cosine embeddings
            sin: Sine embeddings

        Returns:
            Rotated tensor [batch, seq_len, num_kv_heads, head_dim]
        """
        return self._apply_rotary_emb(x, cos, sin)

    def _compute_attention(
        self,
        q: torch.Tensor,
        kv_blocks: list,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention with KV blocks and causal masking.

        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            kv_blocks: List of (K, V) tensor tuples
            hidden_states: Original hidden states (for output shape)

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        batch_size, num_heads, seq_len, head_dim = q.shape

        if not kv_blocks:
            return torch.zeros_like(hidden_states)

        # Concatenate KV blocks
        k_concat = torch.cat([k for k, v in kv_blocks], dim=2)
        v_concat = torch.cat([v for k, v in kv_blocks], dim=2)

        # Get total KV sequence length
        kv_len = k_concat.shape[2]

        # Expand KV for GQA
        k_expanded = k_concat.repeat_interleave(self.bpha_attn.num_groups, dim=1)
        v_expanded = v_concat.repeat_interleave(self.bpha_attn.num_groups, dim=1)

        # Compute attention scores
        scores = torch.einsum('bnqd,bnkd->bnqk', q, k_expanded) * self.bpha_attn.scale

        # Apply causal mask: each query position can only attend to positions up to itself
        # For query position i, it can attend to KV positions 0 to i
        # Create causal mask: [seq_len, kv_len]
        causal_mask = torch.triu(
            torch.ones(seq_len, kv_len, device=q.device, dtype=torch.bool),
            diagonal=max(0, kv_len - seq_len) + 1
        )
        # Mask shape needs to match scores: [batch, num_heads, seq_len, kv_len]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, kv_len]
        scores = scores.masked_fill(causal_mask, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values
        attn_output = torch.einsum('bnqk,bnkd->bnqd', attn_weights, v_expanded)

        # Reshape output
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.bpha_attn.hidden_size)

        # Output projection
        output = self.bpha_attn.o_proj(attn_output)

        return output


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
            block_size=16,  # Matches original BPHA benchmark config for RTX 4060
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
"""
BPHA Attention Layer for Qwen with Grouped Query Attention (GQA) support.

Qwen2.5-3B uses GQA: 16 query heads share 2 KV heads.
Each KV head is shared by 8 query heads (num_groups = num_heads / num_kv_heads).
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn


class BPHAAttention(nn.Module):
    """
    Block-Paged Hybrid Attention for Qwen with GQA support.

    Implements attention with grouped query attention where:
    - Q projection has num_heads outputs
    - K/V projections have num_kv_heads outputs (shared across groups)
    - Each KV head is expanded to serve num_groups query heads

    Args:
        hidden_size: Model hidden dimension
        num_heads: Number of query attention heads
        num_kv_heads: Number of key/value attention heads (for GQA)
        block_size: KV cache block size
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        block_size: int = 16,
        layer_idx: int = 0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.block_size = block_size
        self.layer_idx = layer_idx

        self.head_dim = hidden_size // num_heads
        self.num_groups = num_heads // num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Q projection: full num_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)

        # K/V projections: num_kv_heads * head_dim (GQA)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_blocks: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Forward pass with block-paged KV cache.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            kv_blocks: List of (K_block, V_block) tuples
                Each K_block: [batch, num_kv_heads, block_size, head_dim]
                Each V_block: [batch, num_kv_heads, block_size, head_dim]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]

        if not kv_blocks:
            # No KV cache: return zeros
            return torch.zeros_like(hidden_states)

        # Concatenate KV blocks
        k_concat = torch.cat([k for k, v in kv_blocks], dim=2)  # [batch, num_kv_heads, total_kv_len, head_dim]
        v_concat = torch.cat([v for k, v in kv_blocks], dim=2)  # [batch, num_kv_heads, total_kv_len, head_dim]

        # Expand KV for GQA: repeat each KV head num_groups times
        # [batch, num_kv_heads, kv_len, head_dim] -> [batch, num_heads, kv_len, head_dim]
        k_expanded = k_concat.repeat_interleave(self.num_groups, dim=1)
        v_expanded = v_concat.repeat_interleave(self.num_groups, dim=1)

        # Compute attention scores using einsum for efficiency
        # q: [batch, num_heads, seq_len, head_dim]
        # k_expanded: [batch, num_heads, kv_len, head_dim]
        # scores: [batch, num_heads, seq_len, kv_len]
        scores = torch.einsum('bnqd,bnkd->bnqk', q, k_expanded) * self.scale

        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values
        # attn_weights: [batch, num_heads, seq_len, kv_len]
        # v_expanded: [batch, num_heads, kv_len, head_dim]
        # output: [batch, num_heads, seq_len, head_dim]
        attn_output = torch.einsum('bnqk,bnkd->bnqd', attn_weights, v_expanded)

        # Reshape output
        attn_output = attn_output.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.hidden_size)

        # Output projection
        output = self.o_proj(attn_output)

        return output

    def reshape_for_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reshape K, V for storing in paged cache.

        Args:
            k: [batch, seq_len, hidden_size] (after k_proj)
            v: [batch, seq_len, hidden_size] (after v_proj)

        Returns:
            k_cache: [batch, num_kv_heads, seq_len, head_dim]
            v_cache: [batch, num_kv_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = k.shape

        k_cache = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_cache = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        return k_cache, v_cache

    def __repr__(self) -> str:
        return (
            f"BPHAAttention(hidden_size={self.hidden_size}, "
            f"num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"block_size={self.block_size})"
        )
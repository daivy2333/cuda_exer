"""
BPHA Operator: Core attention operator with block-paged KV cache
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BPHAOperator(nn.Module):
    """
    Block-Paged Hybrid Attention Operator.

    Computes attention over non-contiguous KV blocks using the formula:

        BPHA(Q, K, V) = Σ softmax(Q · K_j^T / √d) · V_j

    where j iterates over logical blocks instead of contiguous memory.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 1, block_size: int = 16):
        """
        Initialize BPHA Operator.

        Args:
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            block_size: KV cache block size
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.head_dim = hidden_dim // num_heads if num_heads > 0 else hidden_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        kv_blocks: List[Tuple[torch.Tensor, torch.Tensor]],
        block_offsets: List[int],
    ) -> torch.Tensor:
        if not kv_blocks:
            return torch.zeros_like(query)

        batch_size, q_len, _ = query.shape
        d_v = kv_blocks[0][1].shape[-1]

        k_concat = torch.cat([k for k, v in kv_blocks], dim=1)
        v_concat = torch.cat([v for k, v in kv_blocks], dim=1)

        output = torch.matmul(torch.softmax(torch.matmul(query, k_concat.transpose(-2, -1)) * self.scale, dim=-1), v_concat)

        return output

    def compute_block_attention(
        self, query: torch.Tensor, k_block: torch.Tensor, v_block: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention for a single block.

        Args:
            query: [batch, q_len, head_dim]
            k_block: [batch, block_size, head_dim]
            v_block: [batch, block_size, head_dim]

        Returns:
            Attention output for this block
        """
        scores = torch.matmul(query, k_block.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v_block)

    def __repr__(self) -> str:
        return (
            f"BPHAOperator(hidden_dim={self.hidden_dim}, "
            f"num_heads={self.num_heads}, block_size={self.block_size})"
        )

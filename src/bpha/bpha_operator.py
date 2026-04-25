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
        """
        Compute BPHA attention.

        Args:
            query: Query tensor [batch, seq_len, hidden_dim]
            kv_blocks: List of (K_block, V_block) tuples
            block_offsets: Starting token offset for each block

        Returns:
            Attention output [batch, seq_len, hidden_dim]
        """
        if not kv_blocks:
            return torch.zeros_like(query)

        batch_size, q_len, _ = query.shape
        outputs = []

        for b in range(batch_size):
            q_b = query[b]
            output_b = torch.zeros_like(q_b)

            for block_idx, ((k_block, v_block), offset) in enumerate(
                zip(kv_blocks, block_offsets)
            ):
                k = k_block[b] if k_block.dim() > 2 else k_block
                v = v_block[b] if v_block.dim() > 2 else v_block

                scores = torch.matmul(q_b, k.transpose(-2, -1)) * self.scale

                attn_weights = F.softmax(scores, dim=-1)

                block_output = torch.matmul(attn_weights, v)

                valid_tokens = k.shape[0]
                end_idx = min(offset + valid_tokens, q_len)
                actual_valid = end_idx - offset
                if actual_valid > 0:
                    output_b[offset:end_idx] += block_output[:actual_valid]

            outputs.append(output_b)

        return torch.stack(outputs, dim=0)

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

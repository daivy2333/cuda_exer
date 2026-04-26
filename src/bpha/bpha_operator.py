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

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 1,
        block_size: int = 16,
        use_cuda_kernel: bool = False,
    ):
        """
        Initialize BPHA Operator.

        Args:
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            block_size: KV cache block size
            use_cuda_kernel: If True, use CUDA fused kernel when available
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.head_dim = hidden_dim // num_heads if num_heads > 0 else hidden_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_cuda_kernel = use_cuda_kernel
        self._cuda_attention = None

        if use_cuda_kernel:
            try:
                from .cuda import paged_attention_fused

                self._cuda_attention = paged_attention_fused
            except ImportError:
                print("Warning: CUDA kernel not available, using Python fallback")
                self.use_cuda_kernel = False

    def forward(
        self,
        query: torch.Tensor,
        kv_blocks: List[Tuple[torch.Tensor, torch.Tensor]],
        block_offsets: List[int],
    ) -> torch.Tensor:
        """
        Forward pass for BPHA.

        Args:
            query: Query tensor [batch, q_len, hidden_dim]
            kv_blocks: List of (k_block, v_block) tuples
            block_offsets: List of block offset positions

        Returns:
            Output tensor [batch, q_len, hidden_dim]
        """
        if not kv_blocks:
            return torch.zeros_like(query)

        # Use CUDA kernel if enabled and tensors are on GPU
        if self.use_cuda_kernel and query.is_cuda:
            return self.forward_cuda(query, kv_blocks, block_offsets)

        # Python fallback
        batch_size, q_len, _ = query.shape
        d_v = kv_blocks[0][1].shape[-1]

        k_concat = torch.cat([k for k, v in kv_blocks], dim=1)
        v_concat = torch.cat([v for k, v in kv_blocks], dim=1)

        output = torch.matmul(
            torch.softmax(
                torch.matmul(query, k_concat.transpose(-2, -1)) * self.scale, dim=-1
            ),
            v_concat,
        )

        return output

    def forward_cuda(
        self,
        query: torch.Tensor,
        kv_blocks: List[Tuple[torch.Tensor, torch.Tensor]],
        block_offsets: List[int],
    ) -> torch.Tensor:
        """
        Forward pass using CUDA fused kernel.

        Converts kv_blocks to the format expected by the CUDA kernel:
        - k_cache: [max_blocks, num_kv_heads, block_size, head_dim]
        - v_cache: [max_blocks, num_kv_heads, block_size, head_dim]
        - block_tables: [batch, max_seq_len / block_size]
        - context_lens: [batch]

        Args:
            query: Query tensor [batch, q_len, hidden_dim]
            kv_blocks: List of (k_block, v_block) tuples
            block_offsets: List of block offset positions

        Returns:
            Output tensor [batch, q_len, hidden_dim]
        """
        batch_size, q_len, hidden_dim = query.shape
        num_blocks = len(kv_blocks)
        block_size = self.block_size
        num_heads = self.num_heads
        head_dim = self.head_dim

        # Reshape query for CUDA kernel: [batch, q_len, hidden_dim] -> [batch, num_heads, q_len, head_dim]
        # For num_heads=1, this is just adding a dimension
        query_4d = query.view(batch_size, q_len, num_heads, head_dim).transpose(1, 2)
        # Now query_4d is [batch, num_heads, q_len, head_dim]

        # Build k_cache and v_cache tensors
        # Each block in kv_blocks has shape [batch, block_size, hidden_dim]
        # We need [max_blocks, num_kv_heads, block_size, head_dim]
        k_blocks = [k for k, v in kv_blocks]
        v_blocks = [v for k, v in kv_blocks]

        # Stack along new dimension: num_blocks blocks
        # k_blocks[i] is [batch, block_size, hidden_dim]
        # We want: [num_blocks, batch, block_size, hidden_dim] -> [num_blocks, num_heads, block_size, head_dim]
        # For simplicity with num_heads=1: reshape each block

        # Stack k and v blocks: shape [num_blocks, batch, block_size, hidden_dim]
        k_stacked = torch.stack(k_blocks, dim=0)
        v_stacked = torch.stack(v_blocks, dim=0)

        # Reshape for CUDA: [num_blocks, num_heads, block_size, head_dim]
        # For num_heads=1, we treat batch as num_kv_heads dimension
        # Since kv_blocks come in with batch dimension, we need to handle this carefully

        # For num_heads=1 case (simple attention):
        # k_stacked: [num_blocks, batch, block_size, hidden_dim]
        # We need: [num_blocks, num_heads=1, block_size, head_dim=hidden_dim]
        # So we add num_heads dimension and remove batch from the cache (assuming single sequence per block)
        # Actually, the CUDA kernel expects k_cache to be shared across batch, with block_tables for indexing

        # For now, handle the single-head, single-batch case
        # The kv_blocks have shape [batch, block_size, hidden_dim] each
        # We need to convert to [num_blocks, num_heads, block_size, head_dim]

        # Taking the first batch element for cache (simplified handling)
        # This works for single-sequence inference
        k_cache = k_stacked[:, 0:1, :, :].reshape(num_blocks, num_heads, block_size, head_dim)
        v_cache = v_stacked[:, 0:1, :, :].reshape(num_blocks, num_heads, block_size, head_dim)

        # Create block_tables: for single sequence, blocks are 0, 1, 2, ...
        max_blocks_per_seq = num_blocks
        block_tables = torch.arange(num_blocks, device=query.device, dtype=torch.int32).unsqueeze(0)
        # block_tables: [1, num_blocks]

        # Compute context length
        context_len = num_blocks * block_size
        context_lens = torch.tensor([context_len], device=query.device, dtype=torch.int32)

        # Call CUDA kernel
        output = self._cuda_attention(
            query=query_4d,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            block_size=block_size,
        )

        # Reshape output back: [batch, num_heads, q_len, head_dim] -> [batch, q_len, hidden_dim]
        output = output.transpose(1, 2).reshape(batch_size, q_len, hidden_dim)

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
            f"num_heads={self.num_heads}, block_size={self.block_size}, "
            f"use_cuda_kernel={self.use_cuda_kernel})"
        )

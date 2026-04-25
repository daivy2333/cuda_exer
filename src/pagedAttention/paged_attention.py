"""
Paged Attention: Attention computation with paged KV Cache
"""

from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F


class PagedAttention:
    """
    Implements attention computation over paged KV Cache.

    Computes attention by accessing KV data stored in physical blocks
    through the BlockTable mapping, maintaining mathematical equivalence
    with standard attention.
    """

    def __init__(self, block_size: int = 128, scale: Optional[float] = None):
        """
        Initialize PagedAttention.

        Args:
            block_size: Block size used in memory manager
            scale: Attention scaling factor (default: 1/sqrt(d_k))
        """
        self.block_size = block_size
        self.scale = scale

    def forward(
        self,
        query: torch.Tensor,
        block_table: "BlockTable",
        seq_id: int,
        num_tokens: int,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute attention with paged KV cache.

        Args:
            query: Query tensor [batch_size, seq_len, d_k]
            block_table: BlockTable mapping logical to physical blocks
            seq_id: Sequence ID for KV lookup
            num_tokens: Total tokens in sequence (including new query)
            past_kv: Optional tuple of (past_k, past_v) tensors

        Returns:
            output: Attention output [batch_size, seq_len, d_v]
            new_kv: Updated (k, v) tensors for this sequence
        """
        batch_size, q_len, d_k = query.shape
        d_v = d_k

        scale = self.scale if self.scale is not None else 1.0 / (d_k**0.5)

        all_k, all_v = self._gather_kv(block_table, seq_id, num_tokens, past_kv)

        scores = torch.matmul(query, all_k.transpose(-2, -1)) * scale

        attn_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attn_weights, all_v)

        return output, (all_k, all_v)

def _gather_kv(
        self,
        block_table: "BlockTable",
        seq_id: int,
        num_tokens: int,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        block_ids = block_table.get_block_ids(seq_id)

        if not block_ids:
            raise ValueError(f"No blocks allocated for seq_id {seq_id}")

        all_k = []
        all_v = []

        for block_id in block_ids:
            block = block_table.physical_blocks[block_id]
            if block.k_data is not None and len(block.tokens) > 0:
                valid_tokens = len(block.tokens)
                k_chunk = torch.from_numpy(block.k_data[:valid_tokens])
                v_chunk = torch.from_numpy(block.v_data[:valid_tokens])
                all_k.append(k_chunk)
                all_v.append(v_chunk)

        if past_kv is not None:
            past_k, past_v = past_kv
            all_k = [past_k.squeeze(0)] + all_k
            all_v = [past_v.squeeze(0)] + all_v

        if all_k:
            all_k = torch.stack(all_k, dim=0)
            all_v = torch.stack(all_v, dim=0)
            all_k = all_k.reshape(-1, all_k.shape[-1]).unsqueeze(0)
            all_v = all_v.reshape(-1, all_v.shape[-1]).unsqueeze(0)
        else:
            d_k = self.block_size
            all_k = torch.zeros(1, num_tokens, d_k)
            all_v = torch.zeros(1, num_tokens, d_k)

        return all_k, all_v


def compare_with_standard_attention(
    paged_attn: PagedAttention,
    block_table: "BlockTable",
    seq_id: int,
    num_tokens: int,
    d_k: int = 64,
    seed: int = 42,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Compare PagedAttention output with standard attention.

    Args:
        paged_attn: PagedAttention instance
        block_table: BlockTable with allocated sequence
        seq_id: Sequence ID
        num_tokens: Number of tokens
        d_k: Key dimension
        seed: Random seed

    Returns:
        max_diff: Maximum difference between outputs
        standard_output: Standard attention output
        paged_output: Paged attention output
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    q = torch.randn(1, 1, d_k)
    k = torch.randn(1, num_tokens, d_k)
    v = torch.randn(1, num_tokens, d_k)

    for i, block_id in enumerate(block_table.get_block_ids(seq_id)):
        block = block_table.physical_blocks[block_id]
        tokens_in_block = min(
            block_table.block_size, num_tokens - i * block_table.block_size
        )
        if block.k_data is not None:
            block.k_data[:tokens_in_block] = k[
                0,
                i * block_table.block_size : i * block_table.block_size
                + tokens_in_block,
            ].numpy()
            block.v_data[:tokens_in_block] = v[
                0,
                i * block_table.block_size : i * block_table.block_size
                + tokens_in_block,
            ].numpy()
            block.tokens = [0] * tokens_in_block

    scale = 1.0 / (d_k**0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    standard_output = torch.matmul(F.softmax(scores, dim=-1), v)

    paged_output, _ = paged_attn.forward(q, block_table, seq_id, num_tokens)

    max_diff = torch.max(torch.abs(standard_output - paged_output)).item()

    return max_diff, standard_output, paged_output

"""
BPHA Compute: Functional implementation of BPHA attention computation
"""

from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
import math


def bpha_forward(
    query: torch.Tensor,
    kv_blocks: List[Tuple[torch.Tensor, torch.Tensor]],
    block_offsets: List[int],
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Forward pass of Block-Paged Hybrid Attention.

    Uses block-wise computation with proper attention semantics.
    For each block, computes attention contribution then accumulates.

    Args:
        query: Query tensor [batch, q_len, d_k]
        kv_blocks: List of (K_block, V_block) tuples
        block_offsets: Starting position for each block
        scale: Attention scale (default: 1/sqrt(d_k))

    Returns:
        Attention output [batch, q_len, d_v]
    """
    if not kv_blocks:
        return torch.zeros_like(query)

    if scale is None:
        d_k = query.shape[-1]
        scale = 1.0 / math.sqrt(d_k)

    batch_size, q_len, d_v = query.shape[0], query.shape[1], kv_blocks[0][1].shape[-1]

    k_concat = torch.cat([k for k, v in kv_blocks], dim=1)
    v_concat = torch.cat([v for k, v in kv_blocks], dim=1)

    scores = torch.matmul(query, k_concat.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v_concat)

    return output


def bpha_backward(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    kv_blocks: List[Tuple[torch.Tensor, torch.Tensor]],
    block_offsets: List[int],
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """
    Backward pass of BPHA attention.

    Args:
        grad_output: Gradient w.r.t. output [batch, q_len, d_v]
        query: Query tensor [batch, q_len, d_k]
        kv_blocks: List of (K_block, V_block) tuples
        block_offsets: Starting position for each block
        scale: Attention scale

    Returns:
        grad_query: Gradient w.r.t. query
        grad_k_blocks: List of gradients for K blocks
        grad_v_blocks: List of gradients for V blocks
    """
    if scale is None:
        d_k = query.shape[-1]
        scale = 1.0 / math.sqrt(d_k)

    batch_size, q_len, d_k = query.shape
    d_v = grad_output.shape[-1]

    grad_query = torch.zeros_like(query)
    grad_k_blocks = []
    grad_v_blocks = []

    for (k_block, v_block), offset in zip(kv_blocks, block_offsets):
        block_size = k_block.shape[-2]
        valid_tokens = min(block_size, q_len - offset)

        if valid_tokens <= 0:
            grad_k_blocks.append(torch.zeros_like(k_block))
            grad_v_blocks.append(torch.zeros_like(v_block))
            continue

        scores = (
            torch.matmul(
                query[:, offset : offset + valid_tokens],
                k_block[:, :valid_tokens].transpose(-2, -1),
            )
            * scale
        )

        attn_weights = F.softmax(scores, dim=-1)

        grad_block_out = grad_output[:, offset : offset + valid_tokens]

        grad_v_block = torch.matmul(attn_weights.transpose(-2, -1), grad_block_out)

        grad_attn = torch.matmul(
            grad_block_out, v_block[:, :valid_tokens].transpose(-2, -1)
        )

        grad_scores = attn_weights * (
            grad_attn - (grad_attn * attn_weights).sum(dim=-1, keepdim=True)
        )
        grad_scores = grad_scores * scale

        grad_k_block = torch.matmul(
            grad_scores.transpose(-2, -1),
            query[:, offset : offset + valid_tokens],
        ).transpose(-2, -1)

        grad_q_block = torch.matmul(grad_scores, k_block[:, :valid_tokens])

        grad_query[:, offset : offset + valid_tokens] += grad_q_block

        full_grad_k = torch.zeros_like(k_block)
        full_grad_k[:, :valid_tokens] = grad_k_block
        grad_k_blocks.append(full_grad_k)

        full_grad_v = torch.zeros_like(v_block)
        full_grad_v[:, :valid_tokens] = grad_v_block
        grad_v_blocks.append(full_grad_v)

    return grad_query, grad_k_blocks, grad_v_blocks


def compute_memory_efficiency(num_tokens: int, block_size: int = 16) -> float:
    """
    Calculate memory efficiency of block-paged storage.

    Args:
        num_tokens: Total number of tokens
        block_size: Size of each block

    Returns:
        Memory efficiency ratio (0.0 to 1.0)
    """
    num_blocks = (num_tokens + block_size - 1) // block_size
    total_capacity = num_blocks * block_size
    waste = total_capacity - num_tokens
    return 1.0 - (waste / total_capacity)

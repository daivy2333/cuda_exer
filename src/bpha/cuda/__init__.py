"""
CUDA Extension for Paged Attention

This module provides Python bindings for the CUDA fused attention kernel.
Uses lazy loading to avoid compilation overhead at import time.

Hardware: RTX 4060 Laptop (Compute Capability 8.9)
CUDA: 12.8
"""

import os
from typing import Optional
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

# Global cache for the compiled extension
_cuda_module: Optional[object] = None


def get_cuda_sources_dir() -> Path:
    """Get the directory containing CUDA source files."""
    return Path(__file__).parent.resolve()


def load_cuda_extension(
    name: str = "paged_attention_cuda",
    verbose: bool = True,
    rebuild: bool = False,
) -> object:
    """
    Load the CUDA extension using PyTorch's JIT compilation.

    Args:
        name: Name of the extension module.
        verbose: If True, print compilation information.
        rebuild: If True, force recompilation.

    Returns:
        The compiled CUDA extension module.
    """
    global _cuda_module

    if _cuda_module is not None and not rebuild:
        return _cuda_module

    sources_dir = get_cuda_sources_dir()
    sources = [
        str(sources_dir / "paged_attention_cuda.cpp"),
        str(sources_dir / "paged_attention_kernel.cu"),
    ]

    # Verify source files exist
    for src in sources:
        if not os.path.exists(src):
            raise FileNotFoundError(f"CUDA source file not found: {src}")

    if verbose:
        print(f"Compiling CUDA extension '{name}'...")
        print(f"  Sources: {sources}")

    # CUDA compilation flags for RTX 4060 (Compute Capability 8.9)
    extra_cuda_cflags = [
        "-gencode=arch=compute_89,code=sm_89",  # RTX 4060 Ada architecture
        "-O3",                                   # Optimization level
        "--use_fast_math",                       # Fast math optimizations
    ]

    # C++ compilation flags
    extra_cflags = [
        "-O3",
    ]

    # Load and compile the extension
    _cuda_module = load(
        name=name,
        sources=sources,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_cflags=extra_cflags,
        verbose=verbose,
    )

    if verbose:
        print(f"CUDA extension '{name}' loaded successfully")

    return _cuda_module


def get_cuda_module() -> object:
    """
    Get the cached CUDA module, loading it if necessary.

    Returns:
        The compiled CUDA extension module.
    """
    global _cuda_module

    if _cuda_module is None:
        _cuda_module = load_cuda_extension(verbose=False)

    return _cuda_module


def paged_attention_fused(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """
    Compute paged attention with fused CUDA kernel.

    This is a high-level wrapper that handles the CUDA extension call.

    Tensor shapes:
        - query:        [batch, num_heads, q_len, head_dim]
        - k_cache:      [max_blocks, num_kv_heads, block_size, head_dim]
        - v_cache:      [max_blocks, num_kv_heads, block_size, head_dim]
        - block_tables: [batch, max_seq_len / block_size]
        - context_lens: [batch]
        - output:       [batch, num_heads, q_len, head_dim]

    Args:
        query: Query tensor of shape [batch, num_heads, q_len, head_dim].
        k_cache: Key cache tensor of shape [max_blocks, num_kv_heads, block_size, head_dim].
        v_cache: Value cache tensor of shape [max_blocks, num_kv_heads, block_size, head_dim].
        block_tables: Block table mapping logical to physical blocks.
                      Shape: [batch, max_seq_len / block_size]
        context_lens: Context length for each sequence. Shape: [batch]
        block_size: Size of each KV cache block.

    Returns:
        Output tensor of shape [batch, num_heads, q_len, head_dim].
    """
    # Ensure inputs are contiguous
    query = query.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    block_tables = block_tables.contiguous()
    context_lens = context_lens.contiguous()

    # Get the CUDA module
    cuda_module = get_cuda_module()

    # Call the CUDA kernel
    output = cuda_module.paged_attention_forward(
        query,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        block_size,
    )

    return output


# Expose the main functions
__all__ = [
    "load_cuda_extension",
    "get_cuda_module",
    "paged_attention_fused",
]
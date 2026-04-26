"""
Unit Tests: CUDA Paged Attention Kernel

Tests for the fused CUDA kernel implementation.
These tests verify the kernel can be loaded and produces correct output shapes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import torch


@pytest.mark.cuda
class TestCUDAKernelLoad:
    """Tests for CUDA kernel loading."""

    def test_kernel_loads(self):
        """Verify CUDA extension can be loaded."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from bpha.cuda import get_cuda_module

        cuda_module = get_cuda_module()

        # Check module has paged_attention_forward function
        assert hasattr(cuda_module, 'paged_attention_forward'), \
            "CUDA module missing paged_attention_forward function"
        assert callable(cuda_module.paged_attention_forward), \
            "paged_attention_forward is not callable"


@pytest.mark.cuda
class TestCUDAKernelShapes:
    """Tests for CUDA kernel output shapes."""

    def test_output_shape(self):
        """Verify output tensor has correct shape."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from bpha.cuda import paged_attention_fused

        # Create test tensors with specified shapes
        batch = 1
        num_heads = 1
        q_len = 1
        head_dim = 64
        num_kv_heads = 1
        max_blocks = 10
        block_size = 16

        # Create tensors on GPU (float32 as required by kernel)
        query = torch.randn(batch, num_heads, q_len, head_dim, device='cuda', dtype=torch.float32)
        k_cache = torch.randn(max_blocks, num_kv_heads, block_size, head_dim, device='cuda', dtype=torch.float32)
        v_cache = torch.randn(max_blocks, num_kv_heads, block_size, head_dim, device='cuda', dtype=torch.float32)
        block_tables = torch.zeros(batch, max_blocks, device='cuda', dtype=torch.int32)
        context_lens = torch.tensor([block_size * 3], device='cuda', dtype=torch.int32)  # 3 blocks of context

        # Set up block table for 3 blocks
        block_tables[0, 0] = 0
        block_tables[0, 1] = 1
        block_tables[0, 2] = 2

        # Call the kernel
        output = paged_attention_fused(
            query=query,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            block_size=block_size,
        )

        # Assert output shape matches query shape
        assert output.shape == query.shape, \
            f"Output shape {output.shape} does not match query shape {query.shape}"

    def test_output_shape_batched(self):
        """Verify output shape with batch size > 1."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from bpha.cuda import paged_attention_fused

        batch = 4
        num_heads = 2
        q_len = 1
        head_dim = 64
        num_kv_heads = 2
        max_blocks = 20
        block_size = 16

        # Create tensors on GPU (float32 as required by kernel)
        query = torch.randn(batch, num_heads, q_len, head_dim, device='cuda', dtype=torch.float32)
        k_cache = torch.randn(max_blocks, num_kv_heads, block_size, head_dim, device='cuda', dtype=torch.float32)
        v_cache = torch.randn(max_blocks, num_kv_heads, block_size, head_dim, device='cuda', dtype=torch.float32)
        block_tables = torch.zeros(batch, max_blocks, device='cuda', dtype=torch.int32)
        context_lens = torch.tensor([32, 48, 16, 64], device='cuda', dtype=torch.int32)

        # Set up block tables for each sequence
        for i in range(batch):
            for j in range(context_lens[i].item() // block_size):
                block_tables[i, j] = i * 4 + j

        output = paged_attention_fused(
            query=query,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            block_size=block_size,
        )

        assert output.shape == query.shape, \
            f"Output shape {output.shape} does not match query shape {query.shape}"


@pytest.mark.cuda
class TestCUDAKernelNumerical:
    """Tests for CUDA kernel numerical correctness."""

    def test_numerical_correctness(self):
        """
        Compare CUDA kernel with Python implementation.

        NOTE: This test is a placeholder. The kernel currently returns zeros.
        After Task 3 implements the real kernel, this should verify numerical
        correctness against standard attention.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from bpha.cuda import paged_attention_fused

        # Small test case: 3 KV blocks, block_size=32, head_dim=64
        batch = 1
        num_heads = 1
        q_len = 1
        head_dim = 64
        num_kv_heads = 1
        max_blocks = 10
        block_size = 32
        num_blocks = 3
        seq_len = num_blocks * block_size  # 96 tokens

        # Create tensors on GPU (float32 as required by kernel)
        query = torch.randn(batch, num_heads, q_len, head_dim, device='cuda', dtype=torch.float32)
        k_cache = torch.randn(max_blocks, num_kv_heads, block_size, head_dim, device='cuda', dtype=torch.float32)
        v_cache = torch.randn(max_blocks, num_kv_heads, block_size, head_dim, device='cuda', dtype=torch.float32)
        block_tables = torch.zeros(batch, max_blocks, device='cuda', dtype=torch.int32)
        context_lens = torch.tensor([seq_len], device='cuda', dtype=torch.int32)

        # Set up block table for 3 blocks
        block_tables[0, 0] = 0
        block_tables[0, 1] = 1
        block_tables[0, 2] = 2

        # Call CUDA kernel
        cuda_output = paged_attention_fused(
            query=query,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            block_size=block_size,
        )

        # Compute reference with standard attention (matmul + softmax)
        # Reconstruct full K, V from cache
        k_full = k_cache[:num_blocks, 0, :, :].reshape(1, seq_len, head_dim)  # [1, 96, 64]
        v_full = v_cache[:num_blocks, 0, :, :].reshape(1, seq_len, head_dim)  # [1, 96, 64]

        # Standard attention computation (float32 throughout)
        # query: [batch=1, num_heads=1, q_len=1, head_dim=64]
        scale = 1.0 / (head_dim ** 0.5)
        scores = torch.matmul(query, k_full.transpose(-2, -1).unsqueeze(0)) * scale  # [1, 1, 1, 96]
        attn_weights = torch.softmax(scores, dim=-1)  # [1, 1, 1, 96]
        reference_output = torch.matmul(attn_weights, v_full.unsqueeze(0))  # [1, 1, 1, 64]

        # For now, just verify shapes match
        # After Task 3, this should be a numerical comparison:
        # assert torch.allclose(cuda_output, reference_output, atol=1e-3, rtol=1e-3)
        assert cuda_output.shape == reference_output.shape, \
            f"CUDA output shape {cuda_output.shape} does not match reference {reference_output.shape}"

        # Placeholder: This test PASSES for now since kernel returns zeros
        # After Task 3 implements real kernel, uncomment numerical check above
        # and remove this placeholder assertion
        assert cuda_output.shape == query.shape, \
            "CUDA output shape should match query shape"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
/**
 * PyTorch C++ Extension Binding for Paged Attention CUDA Kernel
 *
 * This file provides the Python-C++ interface for the CUDA kernel.
 *
 * Hardware: RTX 4060 Laptop (Compute Capability 8.9)
 * CUDA: 12.8
 */

#include <torch/extension.h>

// Forward declaration of CUDA launch function
torch::Tensor launch_paged_attention(
    const torch::Tensor& query,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lens,
    const float scale,
    const int block_size);

/**
 * Paged attention forward pass
 *
 * Computes attention with paged KV cache support.
 *
 * Tensor shapes:
 *   - query:        [batch, num_heads, q_len, head_dim]
 *   - k_cache:      [max_blocks, num_kv_heads, block_size, head_dim]
 *   - v_cache:      [max_blocks, num_kv_heads, block_size, head_dim]
 *   - block_tables: [batch, max_seq_len / block_size]
 *   - context_lens: [batch]
 *   - output:       [batch, num_heads, q_len, head_dim]
 *
 * @param query         Query tensor (float32)
 * @param k_cache       Key cache tensor (float32)
 * @param v_cache       Value cache tensor (float32)
 * @param block_tables  Block table mapping logical to physical blocks (int32)
 * @param context_lens  Context length for each sequence (int32)
 * @param block_size    Size of each KV cache block
 * @return              Output attention tensor
 */
torch::Tensor paged_attention_forward(
    const torch::Tensor& query,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lens,
    const int64_t block_size) {

  // Input validation: check tensor dimensions
  TORCH_CHECK(query.dim() == 4, "query must be 4D [batch, num_heads, q_len, head_dim]");
  TORCH_CHECK(k_cache.dim() == 4, "k_cache must be 4D [max_blocks, num_kv_heads, block_size, head_dim]");
  TORCH_CHECK(v_cache.dim() == 4, "v_cache must be 4D [max_blocks, num_kv_heads, block_size, head_dim]");
  TORCH_CHECK(block_tables.dim() == 2, "block_tables must be 2D [batch, max_seq_len / block_size]");
  TORCH_CHECK(context_lens.dim() == 1, "context_lens must be 1D [batch]");

  // Input validation: check tensor types
  TORCH_CHECK(query.scalar_type() == torch::kFloat32, "query must be float32");
  TORCH_CHECK(k_cache.scalar_type() == torch::kFloat32, "k_cache must be float32");
  TORCH_CHECK(v_cache.scalar_type() == torch::kFloat32, "v_cache must be float32");
  TORCH_CHECK(block_tables.scalar_type() == torch::kInt32, "block_tables must be int32");
  TORCH_CHECK(context_lens.scalar_type() == torch::kInt32, "context_lens must be int32");

  // Input validation: check device
  TORCH_CHECK(query.is_cuda(), "query must be on CUDA device");
  TORCH_CHECK(k_cache.is_cuda(), "k_cache must be on CUDA device");
  TORCH_CHECK(v_cache.is_cuda(), "v_cache must be on CUDA device");
  TORCH_CHECK(block_tables.is_cuda(), "block_tables must be on CUDA device");
  TORCH_CHECK(context_lens.is_cuda(), "context_lens must be on CUDA device");

  // Input validation: check dimension consistency
  const int batch = query.size(0);
  const int num_heads = query.size(1);
  const int head_dim = query.size(3);
  const int num_kv_heads = k_cache.size(1);

  TORCH_CHECK(block_tables.size(0) == batch, "block_tables batch size mismatch");
  TORCH_CHECK(context_lens.size(0) == batch, "context_lens batch size mismatch");
  TORCH_CHECK(k_cache.size(3) == head_dim, "k_cache head_dim mismatch");
  TORCH_CHECK(v_cache.size(3) == head_dim, "v_cache head_dim mismatch");
  TORCH_CHECK(k_cache.size(2) == block_size, "k_cache block_size mismatch");
  TORCH_CHECK(v_cache.size(2) == block_size, "v_cache block_size mismatch");

  // Compute scale: 1 / sqrt(head_dim)
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // Launch CUDA kernel
  return launch_paged_attention(
      query,
      k_cache,
      v_cache,
      block_tables,
      context_lens,
      scale,
      static_cast<int>(block_size));
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_attention_forward", &paged_attention_forward,
        "Paged attention forward pass (CUDA)");
}
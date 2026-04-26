/**
 * Paged Attention CUDA Kernel - Fused Implementation
 *
 * Implements attention computation with online softmax for paged KV cache.
 * Hardware: RTX 4060 Laptop (Compute Capability 8.9)
 * CUDA: 12.8
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

// Kernel configuration
constexpr int THREADS_PER_BLOCK = 128;  // Threads per block
constexpr int WARP_SIZE = 32;           // Warp size for NVIDIA GPUs
constexpr int MAX_HEAD_DIM = 128;       // Maximum supported head dimension

// CUDA error checking macro
#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err = call;                                                     \
    if (err != cudaSuccess) {                                                   \
      const char* err_str = cudaGetErrorString(err);                            \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              err_str);                                                          \
      throw std::runtime_error(std::string("CUDA error: ") + err_str);          \
    }                                                                           \
  } while (0)

/**
 * Block-level reduction for sum
 * Assumes blockDim.x is power of 2 and <= THREADS_PER_BLOCK
 */
__device__ float block_reduce_sum(float val, float* shared_mem) {
  const int lane_idx = threadIdx.x % WARP_SIZE;
  const int warp_idx = threadIdx.x / WARP_SIZE;

  // Warp-level reduction
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }

  // Warp leader writes to shared memory
  if (lane_idx == 0) {
    shared_mem[warp_idx] = val;
  }
  __syncthreads();

  // First warp reads from shared memory and reduces
  if (warp_idx == 0) {
    val = (lane_idx < (blockDim.x / WARP_SIZE)) ? shared_mem[lane_idx] : 0.0f;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      val += __shfl_down_sync(0xffffffff, val, offset);
    }
  }

  return val;
}

/**
 * Fused paged attention kernel with online softmax
 *
 * Each thread block handles one (batch_idx, head_idx, q_pos) combination.
 * Threads cooperate to compute attention over all KV tokens.
 *
 * Algorithm:
 * 1. Load query vector (each thread handles one dimension)
 * 2. For each KV token:
 *    - Compute score = (query · k_token) * scale (block reduction)
 *    - Online softmax update
 * 3. Normalize and write output
 *
 * Grid: (batch * num_heads * q_len) blocks
 * Block: THREADS_PER_BLOCK threads (one thread per dimension)
 *
 * @param output        Output tensor [batch, num_heads, q_len, head_dim]
 * @param query         Query tensor [batch, num_heads, q_len, head_dim]
 * @param k_cache       Key cache [max_blocks, num_kv_heads, block_size, head_dim]
 * @param v_cache       Value cache [max_blocks, num_kv_heads, block_size, head_dim]
 * @param block_tables  Block table [batch, max_blocks]
 * @param context_lens  Context lengths [batch]
 * @param scale         Softmax scale (1 / sqrt(head_dim))
 * @param batch         Batch size
 * @param num_heads     Number of query heads
 * @param num_kv_heads  Number of key/value heads (for GQA)
 * @param q_len         Query sequence length
 * @param head_dim      Head dimension
 * @param block_size    Size of each KV cache block
 * @param max_blocks    Maximum number of blocks per sequence
 */
__global__ void paged_attention_kernel_fused(
    float* __restrict__ output,
    const float* __restrict__ query,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const float scale,
    const int batch,
    const int num_heads,
    const int num_kv_heads,
    const int q_len,
    const int head_dim,
    const int block_size,
    const int max_blocks) {

  // Shared memory for reduction and softmax state
  __shared__ float shared_mem[THREADS_PER_BLOCK / WARP_SIZE];  // For reduction
  __shared__ float shared_max;       // Running max for online softmax
  __shared__ float shared_sum_exp;   // Running sum_exp for online softmax
  __shared__ float shared_correction; // Correction factor from max update

  // Compute which (batch, head, q_pos) this block handles
  const int batch_idx = blockIdx.x / (num_heads * q_len);
  const int remaining = blockIdx.x % (num_heads * q_len);
  const int head_idx = remaining / q_len;
  const int q_pos = remaining % q_len;

  // Each thread handles one dimension
  const int dim_idx = threadIdx.x;

  // Get context length for this sequence
  const int context_len = context_lens[batch_idx];

  // Initialize shared state
  if (threadIdx.x == 0) {
    shared_max = -INFINITY;
    shared_sum_exp = 0.0f;
    shared_correction = 1.0f;
  }
  __syncthreads();

  // Skip if no context to attend to
  if (context_len <= 0) {
    if (dim_idx < head_dim) {
      output[batch_idx * num_heads * q_len * head_dim +
             head_idx * q_len * head_dim +
             q_pos * head_dim + dim_idx] = 0.0f;
    }
    return;
  }

  // For GQA: map query head to KV head
  const int heads_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / heads_per_kv;

  // Compute pointers
  const float* query_ptr = query +
      batch_idx * num_heads * q_len * head_dim +
      head_idx * q_len * head_dim +
      q_pos * head_dim;

  float* output_ptr = output +
      batch_idx * num_heads * q_len * head_dim +
      head_idx * q_len * head_dim +
      q_pos * head_dim;

  // Block table for this sequence
  const int* block_table = block_tables + batch_idx * max_blocks;

  // Load query value for this thread's dimension
  const float query_val = (dim_idx < head_dim) ? query_ptr[dim_idx] : 0.0f;

  // Accumulator for weighted V values
  float acc_val = 0.0f;

  // Number of KV tokens
  const int total_kv_tokens = context_len;

  // Process all KV tokens
  for (int kv_token_idx = 0; kv_token_idx < total_kv_tokens; kv_token_idx++) {
    // Compute block index and intra-block offset
    const int logical_block_idx = kv_token_idx / block_size;
    const int block_offset = kv_token_idx % block_size;

    // Get physical block from block table
    const int physical_block = block_table[logical_block_idx];

    // Skip invalid blocks (shouldn't happen, but safety check)
    if (physical_block < 0) continue;

    // K cache: [max_blocks, num_kv_heads, block_size, head_dim]
    const float* k_ptr = k_cache +
        physical_block * num_kv_heads * block_size * head_dim +
        kv_head_idx * block_size * head_dim +
        block_offset * head_dim;

    // Compute partial dot product (one element per thread)
    float partial_score = (dim_idx < head_dim) ? query_val * k_ptr[dim_idx] : 0.0f;

    // Block-level reduction to get full score
    float full_score = block_reduce_sum(partial_score, shared_mem);

    // Only thread 0 has the final score, broadcast to all
    if (threadIdx.x == 0) {
      shared_mem[0] = full_score * scale;
    }
    __syncthreads();

    const float score = shared_mem[0];

    // Online softmax update
    // Thread 0 manages the shared state
    if (threadIdx.x == 0) {
      const float old_max = shared_max;
      const float new_max = fmaxf(old_max, score);

      // Compute correction factor for previous accumulated values
      const float correction = expf(old_max - new_max);

      // Update shared state
      shared_max = new_max;
      shared_correction = correction;

      // Update sum_exp
      const float exp_score = expf(score - new_max);
      shared_sum_exp = shared_sum_exp * correction + exp_score;
    }
    __syncthreads();

    // All threads read the correction and max
    const float correction = shared_correction;
    const float curr_max = shared_max;

    // V cache: [max_blocks, num_kv_heads, block_size, head_dim]
    const float* v_ptr = v_cache +
        physical_block * num_kv_heads * block_size * head_dim +
        kv_head_idx * block_size * head_dim +
        block_offset * head_dim;

    // Accumulate weighted V value for this thread's dimension
    if (dim_idx < head_dim) {
      const float exp_score = expf(score - curr_max);
      acc_val = acc_val * correction + exp_score * v_ptr[dim_idx];
    }
  }

  // Normalize by sum_exp and write output
  __syncthreads();
  const float sum_exp = shared_sum_exp;
  const float inv_sum_exp = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

  if (dim_idx < head_dim) {
    output_ptr[dim_idx] = acc_val * inv_sum_exp;
  }
}

/**
 * Launch function for paged attention kernel
 *
 * @param query         Query tensor [batch, num_heads, q_len, head_dim]
 * @param k_cache       Key cache [max_blocks, num_kv_heads, block_size, head_dim]
 * @param v_cache       Value cache [max_blocks, num_kv_heads, block_size, head_dim]
 * @param block_tables  Block table [batch, max_seq_len / block_size]
 * @param context_lens  Context lengths [batch]
 * @param scale         Softmax scale
 * @param block_size    Size of each KV cache block
 * @return              Output tensor [batch, num_heads, q_len, head_dim]
 */
torch::Tensor launch_paged_attention(
    const torch::Tensor& query,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lens,
    const float scale,
    const int block_size) {

  // Get tensor dimensions
  const int batch = query.size(0);
  const int num_heads = query.size(1);
  const int q_len = query.size(2);
  const int head_dim = query.size(3);
  const int num_kv_heads = k_cache.size(1);
  const int max_blocks = k_cache.size(0);

  // Allocate output tensor
  auto output = torch::empty_like(query);

  // Configure kernel launch
  // One block per (batch, head, q_pos)
  const int num_blocks = batch * num_heads * q_len;

  // Launch kernel
  paged_attention_kernel_fused<<<num_blocks, THREADS_PER_BLOCK>>>(
      output.data_ptr<float>(),
      query.data_ptr<float>(),
      k_cache.data_ptr<float>(),
      v_cache.data_ptr<float>(),
      block_tables.data_ptr<int>(),
      context_lens.data_ptr<int>(),
      scale,
      batch,
      num_heads,
      num_kv_heads,
      q_len,
      head_dim,
      block_size,
      max_blocks);

  // Check for kernel launch errors
  CUDA_CHECK(cudaGetLastError());

  return output;
}
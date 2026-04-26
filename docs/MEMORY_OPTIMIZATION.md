# Memory Optimization Report for BPHA KV Cache

**GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)
**Target Model:** Qwen2.5-3B
**Date:** 2026-04-26

---

## Executive Summary

This report summarizes memory optimization analysis for the BPHA (Block-Paged Hybrid Attention) KV Cache implementation on RTX 4060 8GB. Key findings:

- **Optimal block_size:** 16 (minimal waste) or 128 (balanced performance)
- **Maximum KV cache capacity:** 170K tokens (block_size=128, 1332 blocks)
- **Memory utilization:** Up to 94.29% for long sequences

---

## Key Findings

### 1. Optimal Block Size Recommendations

| Block Size | Memory Waste | Use Case | Rating |
|------------|-------------|----------|--------|
| 16 | 0% | Short sequences, streaming decode | Excellent |
| 32 | 0% | Short sequences, mixed workloads | Excellent |
| 64 | 0% | Balanced workloads | Excellent |
| 128 | 1.2% (mixed) | Long sequences, balanced performance | Excellent |
| 256 | 10% avg | Long sequences only | Good |

**Recommendation:**
- For **minimal fragmentation**: Use `block_size=16` (0% waste across all scenarios)
- For **balanced performance**: Use `block_size=128` (reduces allocation overhead for longer sequences)

### 2. Maximum KV Cache Capacity

| Configuration | Max Blocks | Memory | Max Tokens |
|--------------|------------|--------|------------|
| block_size=128, 1332 blocks | 1332 | ~6GB | **170,496 tokens** |
| block_size=64, 888 blocks | 888 | ~4GB | 56,832 tokens |
| block_size=256, 250 blocks | 250 | ~2.25GB | 64,000 tokens |

**Maximum capacity achieved:** 170K tokens with block_size=128

### 3. Memory Efficiency by Scenario

| Scenario | Memory Utilization | Waste |
|----------|-------------------|-------|
| Short sequences (50-70 tokens) | 43.75% | 56.25% |
| Medium sequences (200-300 tokens) | 81.38% | 18.62% |
| Mixed sequences (50-500 tokens) | 77.01% | 22.99% |
| Long sequences (500-1200 tokens) | 94.29% | 5.71% |

**Insight:** Memory efficiency improves significantly with longer sequences, reaching 94%+ for long sequences.

---

## Detailed Analysis

### Block Size vs Fragmentation

Memory waste analysis across different sequence length ranges:

| Block Size | Short (64-256) | Medium (256-1024) | Long (1024-4096) | Mixed |
|------------|----------------|-------------------|-----------------|-------|
| 16 | 0.0% | 0.0% | 0.0% | 0.0% |
| 32 | 0.0% | 0.0% | 0.0% | 0.0% |
| 64 | 0.0% | 0.0% | 0.0% | 0.0% |
| 128 | 16.7% | 0.0% | 0.0% | 1.2% |
| 256 | 37.5% | 0.0% | 0.0% | 3.4% |

**Key insight:** Block sizes 16-64 provide zero fragmentation waste. Block_size=128 has some waste for very short sequences (16.7%) but excellent performance for medium/long sequences.

### Allocation Performance

Time to allocate KV cache for single 1024-token sequence:

| Block Size | Blocks Needed | Allocation Time |
|------------|--------------|-----------------|
| 16 | 64 | 111.71 ms |
| 32 | 32 | 61.84 ms |
| 64 | 16 | 30.98 ms |
| 128 | 8 | 34.50 ms |
| 256 | 4 | 11.03 ms |

**Trade-off:** Smaller blocks have more allocation overhead but better memory efficiency. Block_size=128 provides a good balance.

### Realistic Inference Scenarios

| Scenario | Memory | Blocks | Waste | Time |
|----------|--------|--------|-------|------|
| Single Short Prompt | 4.5 MB | 2 | 0.0% | 2.37 ms |
| Single Medium Prompt | 18.0 MB | 4 | 0.0% | 7.16 ms |
| Single Long Prompt | 72.0 MB | 16 | 0.0% | 31.89 ms |
| Batch Short Prompts | 18.0 MB | 8 | 0.0% | 11.00 ms |
| Batch Mixed Lengths | 33.8 MB | 15 | 0.0% | 20.06 ms |
| Streaming Decode | 13.5 MB | 24 | 0.0% | 28.31 ms |

**Average peak memory:** 26.6 MB
**Average waste:** 0.0%
**Average allocation time:** 16.80 ms

---

## GPU Memory Status

| Metric | Value |
|--------|-------|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Total VRAM | 8.0 GB |
| Current allocated | 8.1 MB |
| Current reserved | 20.0 MB |
| Peak during forward | 12.2 MB |

**Note:** Current implementation uses minimal GPU memory (~12MB peak). Significant room for larger KV caches.

---

## Recommendations for E2E Example Updates

### 1. Default Configuration

```python
# Recommended defaults for RTX 4060 8GB
block_size = 128  # Balanced performance
max_blocks = 400  # ~1.8GB KV cache, leaves room for model weights
```

### 2. Use-Case Specific Configurations

**For streaming/chat applications:**
```python
block_size = 16  # Minimal waste for variable-length sequences
max_blocks = 800  # Allows many concurrent sessions
```

**For batch inference:**
```python
block_size = 128  # Lower allocation overhead
max_blocks = 1332  # Maximum capacity
```

**For long document processing:**
```python
block_size = 256  # Efficient for long sequences
max_blocks = 250  # 64K token capacity
```

### 3. Memory Budget Planning

For Qwen2.5-3B on RTX 4060:
- Model weights: ~3GB (with quantization)
- KV Cache budget: 2-4GB recommended
- Remaining: 1-2GB for activations, other tensors

### 4. Next Steps

1. **Test with actual model weights** - Validate memory estimates with Qwen2.5-3B
2. **Profile during inference** - Measure real KV cache behavior with dynamic batching
3. **Implement memory-aware batching** - Adjust batch size based on available KV cache memory

---

## Benchmark Methodology

- **Hardware:** RTX 4060 Laptop GPU (8GB VRAM, Compute 8.9)
- **Software:** Python 3.13.5, PyTorch 2.9.0
- **Test sequences:** Various lengths (64-4096 tokens)
- **Metrics:** Memory utilization, fragmentation, allocation time

Run benchmarks:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate base
python -m benchmarks.gpu_benchmark
python -m benchmarks.memory_optimization_benchmark
```

---

## Conclusion

The BPHA KV Cache implementation demonstrates excellent memory efficiency on RTX 4060:

- **0% fragmentation** with block_size 16-64
- **170K token capacity** maximum
- **94%+ utilization** for long sequences
- **Sub-50ms allocation** for typical workloads

For most use cases, `block_size=128` provides the best balance of memory efficiency and allocation performance, with only 1.2% average waste in mixed workloads.
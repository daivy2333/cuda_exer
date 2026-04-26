# BPHA 基准测试报告

## 测试环境

| 参数 | 值 |
|------|-----|
| **GPU** | NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM, CC 8.9) |
| **CUDA** | 12.8 |
| **Python** | 3.13.5 |
| **PyTorch** | 2.9.0+cu128 |
| **模型** | Qwen2.5-3B-Instruct |
| **日期** | 2026-04-26 |

---

## 1. 内存优化 (Phase 2)

### 1.1 最优 Block Size

| Block Size | 平均浪费率 | 适用场景 | 评级 |
|------------|-----------|----------|------|
| **16** | 3.54% | 短序列、流式解码 | Excellent |
| **64** | 6.25% | 混合负载 | Good |
| **128** | 10.16% | 长序列、均衡性能 | Good |
| **256** | 37.5% (短) | 仅长序列 | Fair |

**推荐:**
- 最小碎片: `block_size=16` (3.54% 平均浪费)
- 均衡性能: `block_size=128` (减少分配开销)

### 1.2 最大 KV Cache 容量

| 配置 | 最大 Blocks | 内存 | 最大 Tokens |
|------|------------|------|-------------|
| block_size=128, max_blocks=1332 | 1332 | ~6GB | **170,496** |
| block_size=128, max_blocks=400 | 400 | ~1.8GB | 51,200 |
| block_size=64, max_blocks=888 | 888 | ~4GB | 56,832 |

**实际配置 (含模型权重):**
```python
block_size = 128
max_blocks = 400  # 留空间给模型 (~3GB)
```

### 1.3 内存利用率

| 场景 | 利用率 | 原因 |
|------|-------|------|
| 短序列 (50-70 tokens) | 43.75% | Block 未填满 |
| 中序列 (200-300 tokens) | 81.38% | 较好填充 |
| 长序列 (500-1200 tokens) | 94.29% | 高效利用 |

---

## 2. 性能对比 (Phase 3)

### 2.1 BPHA vs Standard Attention

| 指标 | BPHA | Standard | 比值 |
|------|------|----------|------|
| 平均延迟 | 0.098 ms | 0.067 ms | **0.67x** |
| 最大吞吐量 | 132K tok/s | - | batch=16 |
| 内存开销 | +0.89 MB | baseline | +3-20% |

**结论:** BPHA 有 ~33% 延迟开销，但提供分页 KV Cache 灵活性。

### 2.2 批处理效率

| Batch Size | 吞吐量 (tok/s) | 效率 |
|------------|---------------|------|
| 1 | 10,893 | 1.00x |
| 4 | 38,861 | 3.57x |
| 8 | 70,863 | 6.51x |
| **16** | **132,052** | **12.11x** |

**最大吞吐量:** 132K tok/s @ batch=16

---

## 3. CUDA Kernel (Phase 4)

### 3.1 实现参数

| 参数 | 值 |
|------|-----|
| Kernel 架构 | sm_89 (Ada Lovelace) |
| 线程数/Block | 128 (fused) / 32 (shared) |
| Tile 大小 | 16x16 |
| 算法 | Online Softmax |

### 3.2 数值精度

| head_dim | Max Difference | 阈值 | 结果 |
|----------|---------------|------|------|
| 16 | 1.45e-7 | 1e-4 | ✅ PASS |
| 64 | 1.79e-7 | 1e-4 | ✅ PASS |
| 128 | 1.19e-7 | 1e-4 | ✅ PASS |

**精度达标:** 所有测试 < 1e-4

### 3.3 延迟对比

| Config | Python (ms) | CUDA (ms) | Speedup |
|--------|-------------|-----------|---------|
| b=1,h=64,bs=16,nb=8 | 0.065 | 0.091 | **0.72x** |
| b=1,h=128,bs=16,nb=8 | 0.069 | 0.075 | **0.93x** |
| b=1,h=64,bs=32,nb=16 | 0.069 | 0.341 | **0.20x** |
| b=4,h=64,bs=32,nb=16 | 0.067 | 0.308 | **0.22x** |
| b=1,h=64,bs=64,nb=32 | 0.059 | 1.257 | **0.05x** |

**平均 Speedup:** 0.34x

### 3.4 性能分析

**CUDA kernel 比 Python 慢的原因:**

| 原因 | 说明 |
|------|------|
| Kernel Launch Overhead | 每次 launch ~5-10μs，小 workload 占主导 |
| PyTorch 优化 | cuBLAS + tensor cores，高度优化 |
| 同步点过多 | 每个 KV token 有 `__syncthreads()` |
| 无 IO-aware 设计 | 需 Flash Attention 风格 tiling |

**CUDA kernel 价值:**
- ✅ Paged memory 灵活性（变长序列）
- ✅ KV Cache 内存效率
- ✅ 批处理内存共享
- ❌ Raw speed (小 workload)

---

## 4. 推荐配置

### 4.1 流式聊天 (内存优先)

```python
block_size = 16   # 最小碎片 (3.54%)
max_blocks = 800  # 支持多并发会话
batch_size = 1-4
use_cuda_kernel = False  # Python 更快
```

### 4.2 批量推理 (吞吐量优先)

```python
block_size = 128  # 减少分配开销
max_blocks = 400  # ~51K tokens
batch_size = 8-16  # 最大吞吐量
use_cuda_kernel = False
```

### 4.3 文档处理 (长序列)

```python
block_size = 128
max_blocks = 400
batch_size = 1    # 长序列优先
use_cuda_kernel = False
```

---

## 5. 运行基准测试

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate base
cd /home/daivy/projects/cuda_exer

# 内存优化测试
python benchmarks/memory_optimization_benchmark.py

# 性能对比测试
python benchmarks/performance_comparison_benchmark.py

# CUDA kernel 测试
python benchmarks/benchmark_cuda_kernel.py

# 全套测试
python -m benchmarks.run_benchmarks
```

---

## 6. 核心发现

### 6.1 内存优化
- Block Size 16 最优 (3.54% 碎片)
- 长序列利用率达 94%
- 最大容量 170K tokens

### 6.2 性能对比
- 批处理收益: batch=16 时 12.11x 提升
- BPHA 开销: 33% 延迟换取内存灵活性
- 最大吞吐量: 132K tok/s

### 6.3 CUDA Kernel
- 数值精度达标 (< 1.2e-7)
- 小 workload 比 Python 慢 (0.34x avg)
- 价值在于内存管理而非 raw speed

---

## 参考

- 内存优化: Section 1
- 性能对比: Section 2
- CUDA Kernel: Section 3
- 算法原理: `docs/ALGORITHM.md`
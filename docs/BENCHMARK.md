# BPHA 基准测试报告

## 测试环境

| 参数 | 值 |
|------|-----|
| **GPU** | NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM) |
| **CUDA** | 12.8 |
| **Python** | 3.13.5 |
| **PyTorch** | 2.9.0 |
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
| 最大吞吐量 | 132K tok/s | - | - |
| 内存开销 | +0.89 MB | baseline | +3-20% |

**结论:** BPHA 有 ~33% 延迟开销，但提供分页 KV Cache 灵活性。

### 2.2 延迟对比详情

| 配置 | BPHA (ms) | Standard (ms) | 比值 |
|------|-----------|---------------|------|
| b=1, s=512, h=64, bs=128 | 0.098 | 0.072 | 0.74x |
| b=4, s=512, h=64, bs=128 | 0.095 | 0.066 | 0.69x |
| b=1, s=1024, h=64, bs=128 | 0.093 | 0.065 | 0.69x |

### 2.3 批处理效率

| Batch Size | 吞吐量 (tok/s) | 效率 |
|------------|---------------|------|
| 1 | 10,893 | 1.00x |
| 4 | 38,861 | 3.57x |
| 8 | 70,863 | 6.51x |
| **16** | **132,052** | **12.11x** |

**最大吞吐量:** 132K tok/s @ batch=16

### 2.4 内存开销

| 配置 | BPHA (MB) | Standard (MB) | 差值 |
|------|-----------|---------------|------|
| b=1, s=512, h=64 | 8.63 | 8.38 | +0.25 MB (+3%) |
| b=4, s=512, h=64 | 10.14 | 9.14 | +1.00 MB (+11%) |
| b=1, s=1024, h=64 | 9.13 | 8.63 | +0.50 MB (+6%) |

---

## 3. 推荐配置

### 3.1 流式聊天

```python
block_size = 16   # 最小碎片
max_blocks = 800  # 支持多并发会话
batch_size = 1-4
```

### 3.2 批量推理

```python
block_size = 128  # 减少分配开销
max_blocks = 400  # ~51K tokens
batch_size = 8-16  # 最大吞吐量
```

### 3.3 文档处理

```python
block_size = 128
max_blocks = 400
batch_size = 1    # 长序列优先
```

---

## 4. 运行基准测试

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate base
cd /home/daivy/projects/cuda_exer

# 内存优化测试
python -m benchmarks.memory_optimization_benchmark

# 性能对比测试
python -m benchmarks.performance_comparison_benchmark

# 全套测试
python -m benchmarks.run_benchmarks
```

---

## 5. 核心发现

1. **批处理收益显著:** batch=16 时吞吐量提升 12.11x
2. **BPHA 开销可接受:** 33% 延迟开销换取内存灵活性
3. **Block Size 影响:** 小 block 减少碎片，大 block 减少开销
4. **分页优势:** 支持动态序列长度，内存利用率高达 94%

---

## 参考

- 内存优化详情: 本报告 Section 1
- 性能对比详情: 本报告 Section 2
- 算法原理: `docs/ALGORITHM.md`
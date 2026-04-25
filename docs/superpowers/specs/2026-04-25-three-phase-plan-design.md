# CUDA Exercise - 三阶段计划设计

## 元信息

- **日期**: 2026-04-25
- **模型**: Qwen2.5-3B-Instruct (替代 TinyLlama 1.1B)
- **GPU**: NVIDIA GeForce RTX 4060 Laptop (8GB)
- **执行方式**: 手动推进（每个阶段完成后确认）
- **方案**: 方案 A - 循序渐进

## 概述

为 CUDA Exercise 项目设计三阶段实现计划，目标是在单 GPU 上验证 BPHA (Block-Paged Hybrid Attention) 在真实模型上的效果，并量化性能和内存指标。

## Phase 1: E2E Model Testing

### 目标

使用 Qwen2.5-3B-Instruct 模型验证 BPHA attention 的端到端正确性。

### 任务

| 任务 | 描述 | 预估时间 |
|------|------|----------|
| 1.1 Qwen 模型适配层 | 创建 model adapter，加载 Qwen2.5-3B-Instruct | 30min |
| 1.2 BPHA attention 替换 | 将模型原生 attention 替换为 BPHA | 60min |
| 1.3 E2E inference 测试 | 运行推理，验证生成输出正确 | 20min |
| 1.4 KV Cache 内存测量 | 测量实际 KV Cache 内存占用 | 15min |

### 验收标准

- 模型生成输出正确（无报错、文本连贯）
- KV Cache 内存指标量化
- 现有 43 个测试继续通过

### 依赖

- Qwen2.5-3B-Instruct 模型已存在于 `model/` 目录
- PyTorch 2.9.0 + CUDA 环境

## Phase 2: Memory Optimization

### 目标

提高 GPU 内存利用率（当前仅使用 ~12MB / 8GB）。

### 任务

| 任务 | 描述 | 预估时间 |
|------|------|----------|
| 2.1 GPU 内存利用率分析 | 分析内存分配瓶颈 | 30min |
| 2.2 Block size 调优 | 测试 block_size=32,64,128 | 45min |
| 2.3 Larger hidden_dim 测试 | 测试 hidden_dim=128,256 | 30min |
| 2.4 KV Cache capacity profiling | 测量实际 vs 理论容量 | 20min |

### 验收标准

- GPU 内存利用率显著提高（目标 > 1GB）
- 找出最优 block_size 配置
- 内存指标量化报告

### 关键文件

- `src/pagedAttention/paged_attention.py`
- `src/memory/memory_tracker.py`

## Phase 3: Performance Comparison

### 目标

量化 BPHA vs 标准 attention 的性能差异。

### 任务

| 任务 | 描述 | 预估时间 |
|------|------|----------|
| 3.1 BPHA vs Standard benchmark | 对比 latency 和 throughput | 40min |
| 3.2 Paged vs Contiguous efficiency | 对比内存利用率 | 30min |
| 3.3 Batching efficiency | 测试不同 batch_size 吞吐量 | 25min |
| 3.4 Final report | 整合数据，更新 CLAUDE.md | 15min |

### 验收标准

- BPHA vs 标准 attention 的延迟对比数据
- Paged vs contiguous 内存效率对比
- 更新 CLAUDE.md benchmark 部分

### 关键文件

- `benchmarks/gpu_benchmark.py`
- `benchmarks/bpha_benchmark.py`

## 整体验收标准

1. **E2E 生成正确**: Qwen 模型使用 BPHA attention 后生成输出正确
2. **内存指标量化**: 实际测量 KV Cache 占用、内存利用率
3. **性能对比数据**: BPHA vs 标准 attention 的延迟对比数据
4. **测试不破坏**: 所有现有 43 个测试继续通过

## 技术栈

- Python 3.13.5
- PyTorch 2.9.0
- CUDA (RTX 4060 8GB)
- Transformers (HuggingFace)

## 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| Qwen 模型太大无法运行 | 使用更小的量化版本或减少 hidden_dim |
| BPHA 替换后推理失败 | 逐步替换 attention layer，单层验证 |
| 内存优化效果不明显 | 先分析瓶颈，针对性优化 |

## 下一步

调用 `writing-plans` skill 为 Phase 1 创建详细实现计划。
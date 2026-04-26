# CUDA Exercise: BPHA Attention 实现

基于论文 "2.5-D Tensor Parallelism with Compiler-aware Paged Attention" 的单卡实现。
已在 RTX 4060 Laptop (8GB VRAM) 上验证。

## 测试环境

| 参数 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (8GB, CC 8.9) |
| CUDA | 12.8 |
| Python | 3.13.5 |
| PyTorch | 2.9.0+cu128 |
| 模型 | Qwen2.5-3B-Instruct |

---

## 核心结果

| 指标 | 值 |
|------|-----|
| 最大吞吐量 | 132K tokens/s (batch=16) |
| BPHA 延迟开销 | +33% vs Standard |
| 内存浪费率 | 3.54% (block_size=16) |
| KV Cache 容量 | 170K tokens (block_size=128) |
| CUDA Kernel 精度 | <1.2e-7 vs Python |
| 测试覆盖 | 54 tests |

---

## 功能列表

| 功能 | 状态 | 说明 |
|------|------|------|
| Block Table | ✅ | 逻辑→物理块映射 |
| Paged Memory Manager | ✅ | KV Cache 分页管理 |
| Paged Attention | ✅ | 分页注意力计算 |
| BPHA Operator | ✅ | 块页混合注意力 |
| Dynamic Batching | ✅ | M/M/1 自适应调度 |
| Blocked Tensor | ✅ | 编译器友好抽象 |
| Qwen Adapter | ✅ | GQA 支持 (16Q/2KV) |
| **CUDA Kernel** | ✅ | 融合 paged attention kernel |
| 2.5-D 并行 | ⚠️ 框架 | 需多卡 NCCL |

---

## 安装

```bash
pip install torch numpy psutil transformers accelerate
```

---

## Quick Start

### 1. Qwen2.5-3B + BPHA

```python
from qwen_adapter import load_qwen_model, replace_attention_with_bpha, KVCacheManager

# 加载模型
model, tokenizer = load_qwen_model("./model/Qwen2.5-3B-Instruct")

# 替换注意力层
replace_attention_with_bpha(model, block_size=128, max_blocks=400)

# 创建 KV Cache 管理器
kv_manager = KVCacheManager(
    num_layers=36,
    num_kv_heads=2,
    head_dim=128,
    block_size=128,
    max_blocks=400,
)

# 推理
input_ids = tokenizer.encode("你好", return_tensors="pt")
output = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0]))
```

### 2. CUDA Kernel (可选)

```python
from bpha import BPHAOperator

# 启用 CUDA kernel
bpha = BPHAOperator(hidden_dim=64, use_cuda_kernel=True)

# 或使用 Python 实现 (默认)
bpha = BPHAOperator(hidden_dim=64)  # use_cuda_kernel=False
```

**CUDA Kernel 参数:**
- `head_dim`: 支持 8, 16, 32, 64, 128
- `block_size`: KV cache block 大小 (16, 32, 64, 128)
- `num_blocks`: KV block 数量

### 3. Paged Attention

```python
from pagedAttention import BlockTable, PagedMemoryManager

bt = BlockTable(block_size=16, num_blocks=100)
bt.allocate(seq_id=1, num_tokens=50)

pmm = PagedMemoryManager(block_size=16, num_blocks=100, hidden_dim=64)
pmm.allocate_sequence(seq_id=1, num_tokens=50)
stats = pmm.get_memory_stats()
```

### 4. Dynamic Batching

```python
from dynamicBatching import AdaptiveBatcher, M1M1Queue

batcher = AdaptiveBatcher(max_batch_size=8, latency_target=0.2)
queue = M1M1Queue(arrival_rate=10.0, service_rate=15.0)
print(f"利用率: {queue.utilization:.2%}")
```

---

## 项目结构

```
cuda_exer/
├── src/
│   ├── pagedAttention/    # 分页注意力
│   ├── bpha/              # BPHA 运算器
│   │   └── cuda/          # CUDA kernel (新增)
│   │       ├── paged_attention_kernel.cu
│   │       ├── paged_attention_cuda.cpp
│   │       └── __init__.py
│   ├── dynamicBatching/   # 动态批处理
│   ├── blockedTensor/     # 编译器抽象
│   ├── memory/            # 内存管理
│   └── qwen_adapter/      # Qwen 模型适配
├── examples/              # E2E 示例
├── benchmarks/            # 基准测试
├── tests/                 # 单元测试 (54 tests)
└── docs/                  # 文档
```

---

## 运行命令

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate base
cd /home/daivy/projects/cuda_exer

# 测试 (54 个测试)
python -m pytest tests/ -v

# 基准测试
python -m benchmarks.run_benchmarks
python benchmarks/benchmark_cuda_kernel.py

# E2E 示例
python examples/example_qwen_bpha.py
```

---

## 推荐配置

| 场景 | Block Size | Max Blocks | Batch Size |
|------|------------|------------|------------|
| 流式聊天 | 16 | 800 | 1-4 |
| 批量推理 | 128 | 400 | 8-16 |
| 文档处理 | 128 | 400 | 1 |

---

## CUDA Kernel Benchmark

| Config | Python (ms) | CUDA (ms) | Speedup |
|--------|-------------|-----------|---------|
| b=1,h=64,bs=16,nb=8 | 0.065 | 0.091 | 0.72x |
| b=1,h=128,bs=16,nb=8 | 0.069 | 0.075 | 0.93x |
| b=1,h=64,bs=32,nb=16 | 0.069 | 0.341 | 0.20x |

**结论:** CUDA kernel 数值精度达标 (<1.2e-7)，但小 workload 下比 Python 慢。

**CUDA kernel 价值:**
- Paged memory 灵活性（变长序列）
- KV Cache 内存效率
- 批处理内存共享

---

## 文档

- [docs/BENCHMARK.md](docs/BENCHMARK.md) - 基准测试报告
- [docs/ALGORITHM.md](docs/ALGORITHM.md) - 算法原理 (中文)

---

## 硬件要求

| 配置 | GPU 内存 | 适用场景 |
|------|----------|----------|
| 最低 | 4GB | CPU 回退 |
| 推荐 | 8GB | 单卡完整测试 |
| 多卡 | 4×24GB | 完整论文复现 |

---

## 多卡功能 (未验证)

以下功能需要多卡环境：

- 2.5-D 张量并行通信
- 跨 GPU All-Reduce
- 分布式 KV Cache

---

## 完成阶段

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | E2E 模型测试 (Qwen2.5-3B) | ✅ |
| Phase 2 | 内存优化 (block_size 调优) | ✅ |
| Phase 3 | 性能对比 (BPHA vs Standard) | ✅ |
| Phase 4 | CUDA Kernel 实现 | ✅ |

---

## License

MIT License

## Citation

```bibtex
@article{pagedattention2024,
  title={2.5-D Tensor Parallelism with Compiler-aware Paged Attention},
}
```
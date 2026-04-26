# CUDA Exercise: BPHA Attention 实现

基于论文 "2.5-D Tensor Parallelism with Compiler-aware Paged Attention" 的单卡实现。
已在 RTX 4060 Laptop (8GB VRAM) 上验证。

## 测试环境

| 参数 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (8GB) |
| CUDA | 12.8 |
| Python | 3.13.5 |
| PyTorch | 2.9.0 |
| 模型 | Qwen2.5-3B-Instruct |

---

## 核心结果

| 指标 | 值 |
|------|-----|
| 最大吞吐量 | 132K tokens/s (batch=16) |
| BPHA 延迟开销 | +33% vs Standard |
| 内存浪费率 | 3.54% (block_size=16) |
| KV Cache 容量 | 170K tokens (block_size=128) |

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

### 2. Paged Attention

```python
from pagedAttention import BlockTable, PagedMemoryManager

# 创建 Block Table
bt = BlockTable(block_size=16, num_blocks=100)
bt.allocate(seq_id=1, num_tokens=50)

# KV Cache 管理器
pmm = PagedMemoryManager(block_size=16, num_blocks=100, hidden_dim=64)
pmm.allocate_sequence(seq_id=1, num_tokens=50)
stats = pmm.get_memory_stats()
```

### 3. Dynamic Batching

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
│   ├── dynamicBatching/   # 动态批处理
│   ├── blockedTensor/     # 编译器抽象
│   ├── memory/            # 内存管理
│   └── qwen_adapter/      # Qwen 模型适配
├── examples/              # E2E 示例
├── benchmarks/            # 基准测试
├── tests/                 # 单元测试
└── docs/                  # 文档
```

---

## 运行命令

```bash
# 测试 (50 个测试)
python -m pytest tests/ -v

# 基准测试
python -m benchmarks.run_benchmarks

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

## License

MIT License

## Citation

```bibtex
@article{pagedattention2024,
  title={2.5-D Tensor Parallelism with Compiler-aware Paged Attention},
  year={2024}
}
```
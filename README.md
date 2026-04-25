# CUDA Exercise: 2.5-D Tensor Parallelism with Paged Attention

A modular implementation and replication study of the paper "2.5-D Tensor Parallelism with Compiler-aware Paged Attention" for single-GPU environments (8GB VRAM).

## Project Overview

This project implements key algorithms from the paper that can run on a single GPU:

- **Paged Attention**: Memory-efficient KV Cache management using block-paging
- **BPHA Operator**: Block-Paged Hybrid Attention computation
- **Dynamic Batching**: Adaptive batch processing based on queuing theory
- **Blocked Tensor**: Compiler-friendly tensor abstraction

## Features

| Feature | Status | Description |
|---------|--------|-------------|
| Block Table | ✅ | Logical-to-physical block mapping |
| Paged Memory Manager | ✅ | KV Cache memory management |
| Paged Attention | ✅ | Attention computation with paged KV |
| BPHA Operator | ✅ | Hybrid attention over non-contiguous blocks |
| Dynamic Batching | ✅ | M/M/1 queue-based adaptive batching |
| Blocked Tensor | ✅ | Compiler-friendly blocked tensor |
| Memory Tracker | ✅ | Memory usage monitoring |
| 2.5-D Parallel | ⚠️ | Framework only (requires multi-GPU) |

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install numpy torch psutil
```

## Quick Start

### 1. Paged Attention

```python
from src.pagedAttention import BlockTable, PagedMemoryManager, PagedAttention

# Create block table
bt = BlockTable(block_size=16, num_blocks=100)

# Allocate sequence
bt.allocate(seq_id=1, num_tokens=50)

# Compute attention
paged_attn = PagedAttention(block_size=16)
output, _ = paged_attn.forward(query, bt, seq_id=1, num_tokens=50)
```

### 2. Dynamic Batching

```python
from src.dynamicBatching import AdaptiveBatcher, M1M1Queue

# Create batcher
batcher = AdaptiveBatcher(max_batch_size=8, latency_target=0.2)

# Analyze queue
queue = M1M1Queue(arrival_rate=10.0, service_rate=15.0)
stats = queue.get_stats()
print(f"Utilization: {stats.utilization:.2%}")
```

### 3. Memory Tracking

```python
from src.memory import MemoryTracker

tracker = MemoryTracker(name="kv_cache")
tracker.allocate(size_bytes=1024, tag="sequence_1")
tracker.print_summary()
```

## Project Structure

```
cuda_exer/
├── src/
│   ├── pagedAttention/     # Paged attention implementation
│   │   ├── block_table.py   # Block mapping
│   │   ├── paged_memory.py  # Memory manager
│   │   └── paged_attention.py # Attention operator
│   ├── bpha/                # Block-Paged Hybrid Attention
│   │   ├── bpha_operator.py  # BPHA operator
│   │   └── bpha_compute.py   # Compute functions
│   ├── dynamicBatching/    # Dynamic batch processing
│   │   ├── adaptive_batcher.py # Adaptive batcher
│   │   └── queue_model.py   # M/M/1 queue model
│   ├── blockedTensor/      # Compiler-friendly tensors
│   │   ├── blocked_tensor.py # Blocked tensor
│   │   └── layout.py        # Layout metadata
│   └── memory/              # Memory management
│       ├── memory_tracker.py # Memory tracking
│       └── allocator.py     # Block allocator
├── examples/                # Usage examples
├── benchmarks/              # Performance benchmarks
├── tests/                   # Unit tests
├── configs/                 # Configuration files
└── docs/                    # Documentation
```

## Examples

See `examples/` directory for detailed examples:

- `example_block_table.py` - Block table usage
- `example_paged_attention.py` - Attention computation
- `example_dynamic_batching.py` - Batch processing
- `example_memory_tracking.py` - Memory monitoring

## Benchmarks

```bash
# Run all benchmarks
python -m benchmarks.run_benchmarks

# Run specific benchmark
python -m benchmarks.memory_benchmark
```

## Testing

```bash
# Run unit tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_block_table.py -v
```

## Documentation

- [Algorithm Principles](docs/algorithm-principles.md) - Theory and design rationale
- [Project Documentation](docs/PROJECT.md) - API reference and architecture

## Hardware Requirements

| Configuration | GPU Memory | Use Case |
|--------------|-----------|----------|
| Minimum | 4GB | Basic algorithms, CPU fallback |
| Recommended | 8GB | Full single-GPU testing |
| Multi-GPU | 4×24GB | Full paper replication |

## Model Support

| Model | Parameters | Memory | Status |
|-------|-----------|--------|--------|
| GPT-2 | 124M | ~1GB | ✅ Tested |
| TinyLlama | 1.1B | ~2.5GB | ✅ Tested |
| OPT-1.3B | 1.3B | ~3GB | ✅ Tested |
| LLaMA-7B | 7B | ~14GB | ⚠️ Requires quantization |

## Algorithms Not Covered (Require Multi-GPU)

- 2.5-D tensor parallelism communication primitives
- Cross-GPU All-Reduce operations
- Multi-GPU KV Cache management
- True distributed runtime

These are implemented as framework/stub code only for architectural completeness.

## License

MIT License

## Citation

```bibtex
@article{pagedattention2024,
  title={2.5-D Tensor Parallelism with Compiler-aware Paged Attention},
  author={},
  year={2024}
}
```
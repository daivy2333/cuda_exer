# Project Documentation

## 1. Overview

**Project**: CUDA Exercise - Paged Attention & 2.5-D Tensor Parallelism
**Goal**: Implement key algorithms from "2.5-D Tensor Parallelism with Compiler-aware Paged Attention" for single-GPU (8GB VRAM).
**Audience**: Researchers and engineers studying LLM inference optimization.

---

## 2. Architecture

### 2.1 Module Dependencies

```
User Applications
       ↓
┌─────────────────────────────────────────────────────────┐
│                    Core Modules                          │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │pagedAttention│  │     bpha    │  │dynamicBatching│    │
│  └──────┬──────┘  └──────┬──────┘  └─────────────┘       │
│         │                │                               │
│         └────────────────┼───────────────────────────── │   │
│                          ↓                               │   │
│  ┌─────────────────────────────────────────────────┐    │   │
│  │              blockedTensor                       │────┘   │
│  └─────────────────────────────────────────────────┘        │
│                          ↓                                 │
│  ┌─────────────────────────────────────────────────┐        │
│  │                    memory                        │        │
│  └─────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Module Summary

| Module | Purpose | File(s) |
|--------|---------|---------|
| `pagedAttention` | KV Cache block-paging | `block_table.py`, `paged_memory.py`, `paged_attention.py` |
| `bpha` | Block-paged hybrid attention | `bpha_operator.py`, `bpha_compute.py` |
| `dynamicBatching` | Adaptive batch processing | `adaptive_batcher.py`, `queue_model.py` |
| `blockedTensor` | Compiler-friendly tensors | `blocked_tensor.py`, `layout.py` |
| `memory` | Memory tracking/allocation | `memory_tracker.py`, `allocator.py` |

---

## 3. API Reference

### 3.1 pagedAttention

```python
from pagedAttention import BlockTable, PagedMemoryManager, PagedAttention

# BlockTable
bt = BlockTable(block_size=16, num_blocks=100)
block_ids = bt.allocate(seq_id=1, num_tokens=50)
bt.free(seq_id=1)

# PagedMemoryManager
pmm = PagedMemoryManager(block_size=16, num_blocks=100, hidden_dim=64)
pmm.allocate_sequence(seq_id=1, num_tokens=50)
pmm.append_tokens(seq_id=1, k_vectors=k, v_vectors=v)
stats = pmm.get_memory_stats()

# PagedAttention
pa = PagedAttention(block_size=16)
output, new_kv = pa.forward(query, block_table, seq_id=1, num_tokens=50)
```

### 3.2 bpha

```python
from bpha import BPHAOperator, bpha_forward, bpha_backward

# Class-based
op = BPHAOperator(hidden_dim=64, num_heads=1, block_size=16)
output = op.forward(query, kv_blocks=[(k1, v1), (k2, v2)], block_offsets=[0, 16])

# Functional
output = bpha_forward(query, [(k1, v1), (k2, v2)], [0, 16])
grad_q, grad_k, grad_v = bpha_backward(grad_out, query, kv_blocks, offsets)
```

### 3.3 dynamicBatching

```python
from dynamicBatching import AdaptiveBatcher, M1M1Queue, Request

# AdaptiveBatcher
batcher = AdaptiveBatcher(max_batch_size=8, latency_target=0.2)
batcher.add_request(Request(...))
batch = batcher.get_batch()
batcher.complete_batch(batch)

# M/M/1 Queue
queue = M1M1Queue(arrival_rate=10.0, service_rate=15.0)
stats = queue.get_stats()
optimal_batch = queue.optimal_batch_size(target_latency=0.2)
```

### 3.4 blockedTensor

```python
from blockedTensor import BlockedTensor, TensorLayout

# BlockedTensor
bt = BlockedTensor(base_shape=(1024, 64), block_size=(16, 16))
bt.set_block(0, block_data)
data = bt.get_block(0)

# TensorLayout
layout = TensorLayout.from_tensor(tensor, block_size=(16, 16))
cache_score = layout.estimate_cache_friendliness()
```

### 3.5 memory

```python
from memory import MemoryTracker, BlockAllocator

# MemoryTracker
tracker = MemoryTracker(name="kv_cache")
tracker.allocate(size_bytes=1024, tag="seq_1")
stats = tracker.get_current_stats()
tracker.print_summary()

# BlockAllocator
alloc = BlockAllocator(num_blocks=100, block_size=4096)
block_ids = alloc.allocate(num_blocks_requested=5)
util = alloc.get_utilization()
```

---

## 4. Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_size` | 16 | Tokens per KV block |
| `num_blocks` | 100 | Maximum physical blocks |
| `hidden_dim` | 64 | Hidden dimension for K/V |
| `max_batch_size` | 8 | Maximum batch size |
| `latency_target` | 0.2s | Target response latency |

See `configs/` directory for YAML files: `default.yaml`, `benchmark.yaml`, `development.yaml`.

---

## 5. Testing

### 5.1 Unit Tests

```bash
python -m pytest tests/ -v
```

| Test File | Coverage |
|-----------|----------|
| `test_block_table.py` | BlockTable allocation, free, mapping |
| `test_bpha.py` | Forward/backward pass, numerical stability |
| `test_dynamic_batching.py` | M/M/1 formulas, batch decisions |
| `test_memory.py` | MemoryTracker, BlockAllocator |

### 5.2 Benchmarks

```bash
python -m benchmarks.run_benchmarks
```

### 5.3 Verification Standards

| Criterion | Target |
|-----------|--------|
| Attention correctness | max_diff < 1e-4 vs standard |
| Memory fragmentation | < 5% |
| Block utilization | > 95% |
| Queue stability | ρ < 0.9 |

---

## 6. Performance

| Operation | Expected Time | Memory |
|-----------|--------------|--------|
| BlockTable alloc (1000) | ~1ms | ~1MB |
| PagedAttention forward | ~10ms | ~100MB |
| Dynamic batch decision | ~0.1ms | negligible |

---

## 7. Limitations

### 7.1 Single-GPU Constraints

The following paper features **cannot** be verified in single-GPU environment:

| Feature | Reason |
|---------|--------|
| 2.5-D Tensor Parallelism | Requires NCCL/GPU interconnect |
| Cross-GPU All-Reduce | Requires multi-GPU hardware |
| Distributed KV Cache | Requires NVLink/PCIe |

These are implemented as framework/stub code for architectural completeness.

### 7.2 Workarounds

| Scenario | Workaround |
|----------|------------|
| Large models | Use smaller models (GPT-2, TinyLlama) |
| Long sequences | Limit to ~2048 tokens |
| Large batch | Reduce batch_size to 8 or less |

---

## 8. Extending

### 8.1 Adding New Modules

1. Create module directory under `src/`
2. Add `__init__.py` with exports
3. Implement core classes/functions
4. Add tests in `tests/`
5. Add examples in `examples/`

### 8.2 Reference Documents

| Document | Purpose |
|----------|---------|
| `algorithm-principles.md` | Theory and algorithm explanations |
| `examples/` | Usage examples for each module |
| `benchmarks/` | Performance measurement scripts |

---

## 9. Glossary

| Term | Definition |
|------|------------|
| KV Cache | Key-Value cache for attention computation |
| Block Table | Mapping from logical blocks to physical blocks |
| BPHA | Block-Paged Hybrid Attention |
| M/M/1 | Queue model: Poisson arrivals, exponential service |
| Utilization (ρ) | System load = arrival_rate / service_rate |

---

## 10. Changelog

| Version | Date | Changes |
|---------|------|---------|
| 0.1.2 | 2026-04-20 | Fixed dead code in adaptive_batcher.py. Consolidated docs (one doc = one responsibility). |
| 0.1.1 | 2026-04-20 | Bugfixes: IndexError in layout.py, deque optimizations, thread safety. |
| 0.1.0 | 2024-04-19 | Initial release |

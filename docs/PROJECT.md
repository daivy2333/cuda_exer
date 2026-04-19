# Project Documentation

## 1. Project Overview

**Project Name**: CUDA Exercise - Paged Attention & 2.5-D Tensor Parallelism

**Objective**: Implement and verify key algorithms from the paper "2.5-D Tensor Parallelism with Compiler-aware Paged Attention" in a single-GPU (8GB VRAM) environment.

**Target Users**: Researchers and engineers studying LLM inference optimization.

---

## 2. Architecture

### 2.1 Module Dependencies

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Applications                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   Examples   │  │  Benchmarks  │  │    Tests     │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                 │                 │                     │
│         ▼                 ▼                 ▼                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                      Core Modules                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │   │
│  │  │pagedAttention│  │     bpha    │  │dynamicBatching│    │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘      │   │
│  │         │                │                │               │   │
│  │         └────────────────┼────────────────┘               │   │
│  │                          ▼                                │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │              blockedTensor (shared)              │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  │                          │                                │   │
│  └──────────────────────────┼────────────────────────────────┘   │
│                             ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    memory (shared)                          │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Core Components

| Module | Purpose | Dependencies |
|--------|---------|-------------|
| `pagedAttention` | KV Cache block-paging | `memory` |
| `bpha` | Block-paged hybrid attention | `blockedTensor` |
| `dynamicBatching` | Adaptive batch processing | None |
| `blockedTensor` | Compiler-friendly tensors | None |
| `memory` | Memory tracking and allocation | None |

---

## 3. Module Specifications

### 3.1 pagedAttention Module

**Purpose**: Memory-efficient KV Cache management using block-paging strategy.

**Components**:

| File | Class/Function | Description |
|------|---------------|-------------|
| `block_table.py` | `BlockTable` | Maps logical sequence blocks to physical memory blocks |
| `block_table.py` | `Block` | Physical block data structure |
| `paged_memory.py` | `PagedMemoryManager` | Manages KV Cache with paged allocation |
| `paged_attention.py` | `PagedAttention` | Attention computation over paged KV |
| `paged_attention.py` | `compare_with_standard_attention` | Verification function |

**Key APIs**:

```python
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

### 3.2 bpha Module

**Purpose**: Block-Paged Hybrid Attention operator implementation.

**Components**:

| File | Class/Function | Description |
|------|---------------|-------------|
| `bpha_operator.py` | `BPHAOperator` | Main attention operator |
| `bpha_compute.py` | `bpha_forward` | Forward pass function |
| `bpha_compute.py` | `bpha_backward` | Backward pass function |
| `bpha_compute.py` | `compute_memory_efficiency` | Efficiency calculation |

**Key APIs**:

```python
# BPHAOperator
op = BPHAOperator(hidden_dim=64, num_heads=1, block_size=16)
output = op.forward(query, kv_blocks=[(k1, v1), (k2, v2)], block_offsets=[0, 16])

# Functional
output = bpha_forward(query, [(k1, v1), (k2, v2)], [0, 16])
grad_q, grad_k, grad_v = bpha_backward(grad_out, query, kv_blocks, offsets)
```

### 3.3 dynamicBatching Module

**Purpose**: Adaptive batch processing based on queuing theory.

**Components**:

| File | Class/Function | Description |
|------|---------------|-------------|
| `adaptive_batcher.py` | `Request` | Request data class |
| `adaptive_batcher.py` | `BatchDecision` | Batch decision data class |
| `adaptive_batcher.py` | `AdaptiveBatcher` | Dynamic batch processor |
| `queue_model.py` | `QueueStats` | Queue statistics data class |
| `queue_model.py` | `M1M1Queue` | M/M/1 queue model |

**Key APIs**:

```python
# AdaptiveBatcher
batcher = AdaptiveBatcher(max_batch_size=8, latency_target=0.2)
batcher.add_request(Request(...))
batch = batcher.get_batch()
batcher.complete_batch(batch)
stats = batcher.get_stats()

# M/M/1 Queue
queue = M1M1Queue(arrival_rate=10.0, service_rate=15.0)
stats = queue.get_stats()
optimal_batch = queue.optimal_batch_size(target_latency=0.2)
```

### 3.4 blockedTensor Module

**Purpose**: Compiler-friendly tensor representation with block structure.

**Components**:

| File | Class/Function | Description |
|------|---------------|-------------|
| `blocked_tensor.py` | `LayoutConstraint` | Layout constraint metadata |
| `blocked_tensor.py` | `BlockedTensor` | Main blocked tensor class |
| `blocked_tensor.py` | `BlockedTensorView` | View with sliced access |
| `layout.py` | `TensorLayout` | Layout metadata container |
| `layout.py` | `ContiguityType` | Contiguity enum |
| `layout.py` | `AccessPattern` | Access pattern enum |

**Key APIs**:

```python
# BlockedTensor
bt = BlockedTensor(base_shape=(1024, 64), block_size=(16, 16))
bt.set_block(0, block_data)
data = bt.get_block(0)
layout_info = bt.get_layout_info()

# TensorLayout
layout = TensorLayout.from_tensor(tensor, block_size=(16, 16))
cache_score = layout.estimate_cache_friendliness()
```

### 3.5 memory Module

**Purpose**: Memory tracking and block allocation.

**Components**:

| File | Class/Function | Description |
|------|---------------|-------------|
| `memory_tracker.py` | `MemoryStats` | Memory statistics data class |
| `memory_tracker.py` | `MemoryTracker` | Memory usage tracker |
| `allocator.py` | `Allocation` | Allocation data class |
| `allocator.py` | `BlockAllocator` | Fixed-size block allocator |

**Key APIs**:

```python
# MemoryTracker
tracker = MemoryTracker(name="kv_cache")
tracker.allocate(size_bytes=1024, tag="seq_1")
tracker.free(size_bytes=1024, tag="seq_1")
stats = tracker.get_current_stats()
tracker.print_summary()

# BlockAllocator
alloc = BlockAllocator(num_blocks=100, block_size=4096)
block_ids = alloc.allocate(num_blocks_requested=5)
alloc.free(block_ids)
util = alloc.get_utilization()
```

---

## 4. Data Flow

### 4.1 Paged Attention Flow

```
Input Query
    │
    ▼
┌─────────────────┐
│  Block Table    │◄────────── Seq ID
│  Lookup         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Get Physical    │
│ Block IDs       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Gather KV from  │
│ Physical Blocks │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Compute         │
│ Attention       │
│ (Standard QK^T) │
└────────┬────────┘
         │
         ▼
    Output
```

### 4.2 Dynamic Batching Flow

```
Requests Arrive
      │
      ▼
┌─────────────────┐
│ M/M/1 Queue     │
│ Model           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Calculate       │
│ Utilization ρ    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Adjust Batch     │
│ Size             │
│ ρ > 0.8 → Max    │
│ ρ < 0.5 → Min    │
└────────┬────────┘
         │
         ▼
    Process Batch
```

---

## 5. Configuration

### 5.1 Default Configurations

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_size` | 16 | Tokens per KV block |
| `num_blocks` | 100 | Maximum physical blocks |
| `hidden_dim` | 64 | Hidden dimension for K/V |
| `max_batch_size` | 8 | Maximum batch size |
| `latency_target` | 0.2s | Target response latency |

### 5.2 Configuration Files

See `configs/` directory for YAML configuration files.

---

## 6. Testing Strategy

### 6.1 Unit Tests

| Test | Coverage |
|------|----------|
| `test_block_table.py` | Allocation, free, mapping |
| `test_paged_memory.py` | Memory stats, append |
| `test_paged_attention.py` | Correctness vs standard |
| `test_bpha.py` | Forward/backward pass |
| `test_dynamic_batching.py` | Queue model calculations |
| `test_blocked_tensor.py` | Layout, indexing |
| `test_memory_tracker.py` | Allocation tracking |

### 6.2 Verification Tests

- **Correctness**: BPHA output matches standard attention
- **Memory Efficiency**: Fragmentation rate < 5%
- **Queue Model**: M/M/1 formulas verified

---

## 7. Performance Metrics

### 7.1 Tracked Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Memory Fragmentation | < 5% | (wasted / total) |
| Block Utilization | > 95% | (used / allocated) |
| Queue Stability | ρ < 0.9 | utilization |
| Attention Correctness | max_diff < 1e-4 | vs standard |

### 7.2 Benchmark Results

Expected single-GPU results on 8GB VRAM:

| Operation | Time | Memory |
|-----------|------|--------|
| BlockTable alloc (1000) | ~1ms | ~1MB |
| PagedAttention forward | ~10ms | ~100MB |
| Dynamic batch decision | ~0.1ms | negligible |

---

## 8. Extension Points

### 8.1 Adding New Modules

1. Create module directory under `src/`
2. Add `__init__.py` with exports
3. Implement core classes/functions
4. Add tests in `tests/`
5. Add examples in `examples/`

### 8.2 Multi-GPU Extensions

The following require multi-GPU environment:

- NCCL-based All-Reduce
- Cross-GPU BlockTable
- 2.5-D collective communication

These are marked with ⚠️ in documentation.

---

## 9. Glossary

| Term | Definition |
|------|------------|
| KV Cache | Key-Value cache for attention computation |
| Block Table | Mapping from logical blocks to physical blocks |
| BPHA | Block-Paged Hybrid Attention |
| M/M/1 | Queue model with Poisson arrivals, exponential service |
| Utilization (ρ) | System load: arrival_rate / service_rate |

---

## 10. References

1. Paper: "2.5-D Tensor Parallelism with Compiler-aware Paged Attention"
2. vLLM: Paged Attention implementation
3. Megatron-LM: Tensor parallelism
4. M/M/1 Queue: Standard queuing theory

---

## 11. Changelog

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2024-04-19 | Initial release |
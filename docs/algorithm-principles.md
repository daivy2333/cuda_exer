# Algorithm Principles

> Theory and design rationale for the core algorithms. For API reference, see `PROJECT.md`.

---

## 1. Paged Attention

### 1.1 Problem

Traditional LLM inference pre-allocates contiguous memory for KV Cache:

```python
token_length = 50
allocated = ((token_length + block_size - 1) // block_size) * block_size  # 64
fragmentation = (allocated - token_length) / allocated  # ~22%
```

For long sequences, this causes severe memory waste.

### 1.2 Solution: Block Paging

Borrowed from OS virtual memory paging:

```
Logical View:                    Physical View:
Token [0-49]     BlockTable      Physical blocks
    ↓            (mapping)           ↓
block 0 →──────────────────────────→ Block #2: [0-15]
block 1 →──────────────────────────→ Block #5: [16-31]
block 2 →──────────────────────────→ Block #7: [32-47]
                                      Block #9: [48-49][...]
```

### 1.3 Memory Efficiency

```
Fragmentation η = 1 - (L̄ / L_max) - (B / L_max)

L̄=avg length, L_max=max length, B=block size
```

| Sequence | Contiguous | Paged |
|----------|------------|-------|
| Short (50-70) | 25-35% | <5% |
| Medium (200-300) | 8-15% | <3% |
| Long (1000-1500) | 2-5% | <1% |

---

## 2. BPHA (Block-Paged Hybrid Attention)

### 2.1 Standard vs Block-wise

**Standard Attention**:
```
Attention(Q, K, V) = softmax(Q · K^T / √d) · V
```

**BPHA** (block-wise decomposition):
```
BPHA(Q, K, V) = Σ_j softmax(Q · K_j^T / √d) · V_j
                 ↑
              j iterates over blocks
```

### 2.2 Implementation

```python
def bpha_forward(query, kv_blocks, block_offsets, scale=None):
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])
    
    output = torch.zeros_like(query)
    for (k_block, v_block), offset in zip(kv_blocks, block_offsets):
        scores = torch.matmul(query, k_block.transpose(-2, -1)) * scale
        attn_weights = F.softmax(scores, dim=-1)
        block_output = torch.matmul(attn_weights, v_block)
        valid_tokens = k_block.shape[-2]
        output[:, offset:offset + valid_tokens] += block_output
    return output
```

### 2.3 Mathematical Equivalence

BPHA maintains correctness because:
- softmax is applied per-block with correct positional indices
- linear weighted sum is associative: `(a+b)+c = a+(b+c)`

The `block_offsets` parameter ensures tokens map to correct positions.

---

## 3. Dynamic Batching

### 3.1 Problem

LLM inference requests are bursty:
```
Time 1: 10 requests → 8 process, 2 wait → latency spike
Time 2: 2 requests → 2 process, 6 GPU idle → waste
```

Static batching cannot adapt to dynamic load.

### 3.2 M/M/1 Queue Model

| Parameter | Meaning |
|-----------|---------|
| λ (arrival_rate) | Request arrival rate |
| μ (service_rate) | Service completion rate |
| ρ = λ/μ | System utilization |

**Core formulas**:
```python
class M1M1Queue:
    @property
    def utilization(self):
        return self.arrival_rate / self.service_rate
    
    def avg_queue_length(self):      # L_q
        rho = self.utilization
        return (rho ** 2) / (1 - rho)
    
    def avg_response_time(self):     # W
        return self.avg_queue_length() / self.arrival_rate + 1 / self.service_rate
```

### 3.3 Adaptive Strategy

```python
def decide_batch_size(self):
    rho = self.lambda_rate / self.mu_rate
    if rho > 0.8:      # High load → maximize throughput
        return self.max_batch_size
    elif rho > 0.5:    # Medium load
        return (self.max + self.min) // 2
    else:              # Low load → minimize latency
        return self.min_batch_size
```

**Rationale**: When ρ approaches 1, queue length explodes. Increasing batch_size amortizes queueing overhead. When ρ is low, process quickly to minimize waiting.

---

## 4. Blocked Tensor

### 4.1 Design Goal

Provide explicit memory layout metadata to compilers for optimization.

```python
@dataclass
class LayoutConstraint:
    alignment: int = 16
    contiguity: str = 'non-contiguous'
    access_pattern: str = 'blocked-random'  # Compiler hint
```

### 4.2 Compiler Benefits

1. **Access pattern declaration**: Compiler selects vectorization strategy
2. **Block-level prefetch**: Can prefetch entire blocks, not just elements
3. **Loop blocking**: Transform loops based on known block_size for cache locality

---

## 5. FAQ

### Q: Paged Attention vs traditional Attention?

Traditional Attention pre-allocates contiguous memory → memory fragmentation. Paged Attention uses BlockTable for non-contiguous physical storage → logical continuity + memory efficiency.

### Q: How does BPHA maintain correctness?

Mathematical equivalence: `softmax(Q·K^T)·V = Σ softmax(Q·K_j^T)·V_j` when `block_offsets` provides correct positional mapping.

### Q: How does dynamic batching balance latency/throughput?

M/M/1 queue model estimates system load (ρ). High ρ → increase batch_size (amortize overhead). Low ρ → decrease batch_size (minimize wait).

### Q: What about 2.5-D Tensor Parallelism?

Requires multi-GPU with NCCL. Implemented as framework stubs only. Cannot verify in single-GPU environment.

# 算法原理

> 基于 "2.5-D Tensor Parallelism with Compiler-aware Paged Attention" 论文的算法解析。

---

## 测试环境

| 参数 | 值 |
|------|-----|
| **GPU** | NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM) |
| **CUDA** | 12.8 |
| **Python** | 3.13.5 |
| **PyTorch** | 2.9.0 |
| **模型** | Qwen2.5-3B-Instruct |

---

## 1. 核心痛点

| 问题 | 传统方案 | 碎片率 |
|------|----------|--------|
| **内存墙** | 连续 KV Cache 预分配 | 40-60% |
| **通信墙** | 1-D 张量并行 O(p) | - |
| **编译器冲突** | 动态序列长度 → 无法融合优化 | - |

**本方案:** 2.5-D 并行 + 分页注意力 + 编译器抽象层

---

## 2. 三大创新

### 2.1 Block-Paged Hybrid Attention (BPHA)

**问题:** 标准 PagedAttention 在多卡下 KV Block 分散，频繁跨设备访问。

**解决方案:**
- 三维处理器网格 `[q, q, d]`
- 块级通信原语 (Block-wise All-Gather)
- 粗粒度块状通信替代细粒度随机访问

**计算流程:**
```
1. 本地加载 → 按 BlockTable 加载 K/V 块
2. 局部计算 → Q · K^T 得注意力分数
3. 行方向聚合 → All-Gather 分片分数
4. Softmax → 计算注意力权重
5. 列方向聚合 → All-Reduce 最终输出
```

### 2.2 编译器感知抽象层

**Blocked Tensor:**
```python
@dataclass
class BlockedTensor:
    base_shape: Tuple[int, int]     # 逻辑形状
    block_size: Tuple[int, int]     # 物理块大小
    block_map: Dict[int, int]       # 逻辑→物理映射
    layout_constraints: Layout      # 编译器提示
```

**并行原语库:**
| 原语 | 功能 |
|------|------|
| Paged MatMul (PMM) | 非连续内存矩阵乘 |
| Blocked Softmax (BS) | 任意大小块 Softmax |
| Distributed All-Reduce | 小消息优化归约 |

**优化策略:**
- 通信-计算重叠 (流水线)
- 算子融合 (减少中间写回)

### 2.3 动态调度器

**M/M/1 排队模型:**
```python
ρ = λ / μ  # 系统负载

def decide_batch_size():
    if ρ > 0.8:  return max_batch  # 高负载→最大化吞吐
    elif ρ > 0.5: return medium    # 中负载
    else: return min_batch         # 低负载→最小延迟
```

**热度感知合并:**
- 优先打包访问相同 "热块" (系统提示词) 的请求
- Cross-Request KV Sharing → 减少 30-50% 计算开销

---

## 3. 理论增益

| 指标 | 传统 | 本方案 | 提升 |
|------|------|--------|------|
| 通信复杂度 | O(p) | O(p^{2/3}) | ~15% 时间占比 |
| 内存碎片率 | 40-60% | **< 5%** | 72.5% 内存节省 |
| 吞吐量提升 | baseline | **3.4x** | A6000 4卡验证 |

---

## 4. 实验结果 (论文 A6000 4卡)

| 模型 | 系统 | 吞吐量 | TTFT | 显存 | SM 利用率 |
|------|------|--------|------|------|----------|
| Qwen3-32B-AWQ | Megatron-LM | 312 tok/s | 350ms | 38.2GB | 65.3% |
| | vLLM | 485 tok/s | 280ms | 22.7GB | 72.1% |
| | **本方案** | **899 tok/s** | **185ms** | **18.3GB** | **91.5%** |

---

## 5. 本项目实现 (RTX 4060 单卡)

| 组件 | 实现状态 | 说明 |
|------|----------|------|
| Block Table | ✅ 完整 | 逻辑→物理块映射 |
| Paged Memory Manager | ✅ 完整 | KV Cache 分页管理 |
| Paged Attention | ✅ 完整 | 分页注意力计算 |
| BPHA Operator | ✅ 完整 | 块页混合注意力 |
| Dynamic Batching | ✅ 完整 | M/M/1 自适应 |
| Blocked Tensor | ✅ 完整 | 编译器抽象 |
| **2.5-D 并行** | ⚠️ 框架 | 需多卡 NCCL |
| **分布式 All-Reduce** | ⚠️ 框架 | 需 GPU 互联 |

---

## 6. 关键公式

### 6.1 内存碎片率

```
η = 1 - (L̄ / L_max) - (B / L_max)

L̄ = 平均序列长度
L_max = 最大预分配长度
B = block_size
```

### 6.2 BPHA 等价性

```
标准: Attention(Q, K, V) = softmax(Q · K^T / √d) · V

BPHA: = Σ_j softmax(Q · K_j^T / √d) · V_j
       ↑ block_offsets 保证位置正确
```

### 6.3 排队论

```
利用率: ρ = λ / μ
平均队列长度: L_q = ρ² / (1 - ρ)
平均响应时间: W = L_q / λ + 1/μ
```

---

## 7. Block Size 选择

| Block Size | 适用场景 | 碎片率 |
|------------|----------|--------|
| 16 | 短序列、流式 | 0-3.5% |
| 64 | 混合负载 | 5-10% |
| 128-256 | 长序列 | 5-10% |

**论文建议:** Block Size 128-256 时缓存命中率最高。

---

## 8. FAQ

**Q: 分页 vs 传统注意力?**
A: 传统预分配连续内存 → 碎片率高。分页用 BlockTable → 逻辑连续 + 物理非连续 → 高效。

**Q: BPHA 如何保证正确性?**
A: `block_offsets` 提供位置映射，softmax 按块应用，线性加权可分解。

**Q: 动态批处理如何平衡?**
A: 高负载 (ρ>0.8) → 大 batch 提吞吐。低负载 → 小 batch 降延迟。

---

## 参考

- 论文: "2.5-D Tensor Parallelism with Compiler-aware Paged Attention"
- 基准测试: `docs/BENCHMARK.md`
- 代码: `src/bpha/`, `src/pagedAttention/`
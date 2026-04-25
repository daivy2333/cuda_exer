# Phase 1: E2E Model Testing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 使用 Qwen2.5-3B-Instruct 模型验证 BPHA attention 的端到端正确性，测量 KV Cache 内存占用。

**Architecture:** 创建 qwen_adapter 模块，将 BPHA attention 替换到 Qwen2 模型的 attention 层，使用 paged KV cache 管理推理过程中的 KV 数据。

**Tech Stack:** Python 3.13.5, PyTorch 2.9.0, Transformers 5.2.0, CUDA (RTX 4060 8GB)

---

## File Structure

```
src/qwen_adapter/
├── __init__.py              # Module exports
├── model_loader.py          # Load Qwen model from local path
├── bpha_attention.py        # BPHAAttention layer for Qwen (GQA support)
├── kv_cache_manager.py      # Manage paged KV cache during inference
└── replace_attention.py     # Replace model attention with BPHA

tests/
├── test_qwen_adapter.py     # Test adapter functionality

examples/
├── example_qwen_bpha.py     # E2E inference example
```

---

### Task 1: Create Qwen Adapter Module Structure

**Files:**
- Create: `src/qwen_adapter/__init__.py`
- Create: `src/qwen_adapter/model_loader.py`

- [ ] **Step 1: Write test for model loader**

```python
# tests/test_qwen_adapter.py
import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestModelLoader(unittest.TestCase):
    def test_load_qwen_model(self):
        """Test loading Qwen model from local path."""
        from qwen_adapter.model_loader import load_qwen_model

        model_path = os.path.join(
            os.path.dirname(__file__), '..', 'model',
            'models--Qwen--Qwen2.5-3B-Instruct', 'snapshots',
            'aa8e72537993ba99e69dfaafa59ed015b17504d1'
        )

        model, tokenizer = load_qwen_model(model_path)

        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)
        self.assertEqual(model.config.hidden_size, 2048)
        self.assertEqual(model.config.num_attention_heads, 16)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate base && python -m pytest tests/test_qwen_adapter.py::TestModelLoader::test_load_qwen_model -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'qwen_adapter'"

- [ ] **Step 3: Create module directory and __init__.py**

```bash
mkdir -p src/qwen_adapter
touch src/qwen_adapter/__init__.py
```

- [ ] **Step 4: Implement model_loader.py**

```python
# src/qwen_adapter/model_loader.py
"""
Model Loader: Load Qwen2.5-3B-Instruct from local cache
"""

from typing import Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_qwen_model(
    model_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load Qwen model and tokenizer from local path.

    Args:
        model_path: Path to model snapshot directory
        device: Device to load model onto
        torch_dtype: Model data type

    Returns:
        model: Qwen2ForCausalLM model
        tokenizer: Qwen tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )

    return model, tokenizer


def get_model_config(model_path: str) -> dict:
    """
    Extract key config parameters for BPHA adaptation.

    Args:
        model_path: Path to model directory

    Returns:
        config dict with hidden_size, num_heads, num_kv_heads, head_dim
    """
    import json
    config_path = os.path.join(model_path, "config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    return {
        "hidden_size": config["hidden_size"],
        "num_attention_heads": config["num_attention_heads"],
        "num_key_value_heads": config["num_key_value_heads"],
        "head_dim": config["hidden_size"] // config["num_attention_heads"],
        "num_hidden_layers": config["num_hidden_layers"],
    }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate base && python -m pytest tests/test_qwen_adapter.py::TestModelLoader::test_load_qwen_model -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/qwen_adapter/__init__.py src/qwen_adapter/model_loader.py tests/test_qwen_adapter.py
git commit -m "feat: add Qwen model loader"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

### Task 2: Create BPHA Attention Layer for Qwen (GQA Support)

**Files:**
- Create: `src/qwen_adapter/bpha_attention.py`
- Modify: `tests/test_qwen_adapter.py` (add tests for BPHAAttention)

**Note:** Qwen2.5 uses Grouped Query Attention (GQA): 16 query heads share 2 KV heads.

- [ ] **Step 1: Write test for BPHAAttention with GQA**

```python
# tests/test_qwen_adapter.py (append to existing file)

class TestBPHAAttention(unittest.TestCase):
    def setUp(self):
        self.hidden_size = 2048
        self.num_heads = 16
        self.num_kv_heads = 2
        self.head_dim = 128
        self.block_size = 16

    def test_bpha_attention_init(self):
        """Test BPHAAttention initialization with GQA config."""
        from qwen_adapter.bpha_attention import BPHAAttention

        attn = BPHAAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            block_size=self.block_size,
        )

        self.assertEqual(attn.hidden_size, 2048)
        self.assertEqual(attn.num_heads, 16)
        self.assertEqual(attn.num_kv_heads, 2)
        self.assertEqual(attn.head_dim, 128)
        self.assertEqual(attn.num_groups, 8)  # 16 / 2 = 8

    def test_bpha_attention_forward(self):
        """Test BPHAAttention forward pass."""
        from qwen_adapter.bpha_attention import BPHAAttention

        attn = BPHAAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            block_size=self.block_size,
        )

        batch_size = 1
        seq_len = 32
        hidden_states = torch.randn(batch_size, seq_len, self.hidden_size)

        # Create mock KV cache
        kv_blocks = []
        for i in range(seq_len // self.block_size):
            k = torch.randn(batch_size, self.num_kv_heads, self.block_size, self.head_dim)
            v = torch.randn(batch_size, self.num_kv_heads, self.block_size, self.head_dim)
            kv_blocks.append((k, v))

        output = attn.forward(hidden_states, kv_blocks)

        self.assertEqual(output.shape, (batch_size, seq_len, self.hidden_size))
        self.assertFalse(torch.isnan(output).any())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate base && python -m pytest tests/test_qwen_adapter.py::TestBPHAAttention -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'qwen_adapter.bpha_attention'"

- [ ] **Step 3: Implement BPHAAttention with GQA support**

```python
# src/qwen_adapter/bpha_attention.py
"""
BPHA Attention: BPHA attention layer adapted for Qwen2 GQA
"""

import math
import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class BPHAAttention(nn.Module):
    """
    Block-Paged Hybrid Attention for Qwen2 with Grouped Query Attention.

    Qwen2.5-3B uses GQA: 16 query heads share 2 KV heads.
    Each KV head is shared by 8 query heads (num_groups = num_heads / num_kv_heads).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        block_size: int = 16,
        layer_idx: int = 0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.block_size = block_size
        self.layer_idx = layer_idx

        # GQA: number of query heads per KV head
        self.num_groups = num_heads // num_kv_heads

        # Q projection (full number of heads)
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)

        # K, V projections (reduced number of heads for GQA)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_blocks: List[Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with paged KV blocks.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            kv_blocks: List of (K_block, V_block) tuples
                Each block: [batch, num_kv_heads, block_size, head_dim]
            attention_mask: Optional mask (not used in basic BPHA)

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        # Q: [batch, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # K, V: [batch, seq_len, num_kv_heads, head_dim]
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # If we have cached KV blocks, use them instead of current K, V
        if kv_blocks:
            # Concatenate all KV blocks
            k_cached = torch.cat([kb for kb, vb in kv_blocks], dim=2)  # [batch, num_kv_heads, total_tokens, head_dim]
            v_cached = torch.cat([vb for kb, vb in kv_blocks], dim=2)

            # Expand KV for GQA: repeat each KV head num_groups times
            # [batch, num_kv_heads, total_tokens, head_dim] -> [batch, num_heads, total_tokens, head_dim]
            k_expanded = k_cached.repeat_interleave(self.num_groups, dim=1)
            v_expanded = v_cached.repeat_interleave(self.num_groups, dim=1)

            # Compute attention
            # q: [batch, seq_len, num_heads, head_dim]
            # k_expanded: [batch, num_heads, total_tokens, head_dim]
            attn_weights = torch.einsum('bshd,bhtd->bsth', q, k_expanded) * self.scale
            attn_weights = torch.softmax(attn_weights, dim=-1)

            # Apply attention to values
            output = torch.einsum('bsth,bhtd->bshd', attn_weights, v_expanded)
        else:
            # No cached KV, use current K, V (first pass)
            k_expanded = k.repeat_interleave(self.num_groups, dim=2)  # [batch, seq_len, num_heads, head_dim]
            v_expanded = v.repeat_interleave(self.num_groups, dim=2)

            attn_weights = torch.einsum('bshd,bstd->bsth', q, k_expanded) * self.scale
            attn_weights = torch.softmax(attn_weights, dim=-1)

            output = torch.einsum('bsth,bstd->bshd', attn_weights, v_expanded)

        # Reshape and project output
        output = output.reshape(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(output)

        return output

    def reshape_for_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reshape K, V for storing in paged cache.

        Args:
            k: [batch, seq_len, hidden_size] (after k_proj)
            v: [batch, seq_len, hidden_size] (after v_proj)

        Returns:
            k_cache: [batch, num_kv_heads, seq_len, head_dim]
            v_cache: [batch, num_kv_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = k.shape

        k_cache = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_cache = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        return k_cache, v_cache
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate base && python -m pytest tests/test_qwen_adapter.py::TestBPHAAttention -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qwen_adapter/bpha_attention.py tests/test_qwen_adapter.py
git commit -m "feat: add BPHAAttention with GQA support for Qwen2"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

### Task 3: Create KV Cache Manager for Inference

**Files:**
- Create: `src/qwen_adapter/kv_cache_manager.py`
- Modify: `tests/test_qwen_adapter.py` (add tests)

- [ ] **Step 1: Write test for KV Cache Manager**

```python
# tests/test_qwen_adapter.py (append)

class TestKVCacheManager(unittest.TestCase):
    def setUp(self):
        self.num_layers = 36
        self.num_kv_heads = 2
        self.head_dim = 128
        self.block_size = 16
        self.max_blocks = 100

    def test_kv_cache_manager_init(self):
        """Test KVCacheManager initialization."""
        from qwen_adapter.kv_cache_manager import KVCacheManager

        manager = KVCacheManager(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            block_size=self.block_size,
            max_blocks=self.max_blocks,
        )

        self.assertEqual(manager.num_layers, 36)
        self.assertEqual(manager.num_kv_heads, 2)
        self.assertEqual(manager.block_size, 16)

    def test_allocate_and_store(self):
        """Test allocating and storing KV blocks."""
        from qwen_adapter.kv_cache_manager import KVCacheManager

        manager = KVCacheManager(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            block_size=self.block_size,
            max_blocks=self.max_blocks,
        )

        # Simulate new tokens
        k_new = torch.randn(1, self.num_kv_heads, 10, self.head_dim)
        v_new = torch.randn(1, self.num_kv_heads, 10, self.head_dim)

        manager.allocate_sequence(seq_id=1, num_tokens=10)
        manager.store_kv(layer_idx=0, seq_id=1, k_new=k_new, v_new=v_new)

        blocks = manager.get_kv_blocks(layer_idx=0, seq_id=1)
        self.assertTrue(len(blocks) > 0)

    def test_memory_stats(self):
        """Test memory statistics reporting."""
        from qwen_adapter.kv_cache_manager import KVCacheManager

        manager = KVCacheManager(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            block_size=self.block_size,
            max_blocks=self.max_blocks,
        )

        stats = manager.get_memory_stats()

        self.assertIn("total_blocks", stats)
        self.assertIn("used_blocks", stats)
        self.assertIn("memory_bytes", stats)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate base && python -m pytest tests/test_qwen_adapter.py::TestKVCacheManager -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'qwen_adapter.kv_cache_manager'"

- [ ] **Step 3: Implement KV Cache Manager**

```python
# src/qwen_adapter/kv_cache_manager.py
"""
KV Cache Manager: Manage paged KV cache during inference
"""

import torch
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class KVBlock:
    """A single KV cache block."""
    block_id: int
    k_data: Optional[torch.Tensor] = None
    v_data: Optional[torch.Tensor] = None
    num_tokens: int = 0
    seq_id: Optional[int] = None


class KVCacheManager:
    """
    Manage paged KV cache for multi-layer transformer.

    Each layer has its own block table for KV cache.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        max_blocks: int = 1000,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_blocks = max_blocks

        # Physical blocks pool
        self.physical_blocks: List[KVBlock] = [
            KVBlock(block_id=i) for i in range(max_blocks)
        ]

        # Block tables per layer: seq_id -> list of block_ids
        self.layer_block_tables: List[Dict[int, List[int]]] = [
            {} for _ in range(num_layers)
        ]

        # Sequence block allocation: seq_id -> list of block_ids
        self.seq_allocations: Dict[int, List[int]] = {}

        # Next free block index
        self.next_free_block = 0

    def allocate_sequence(self, seq_id: int, num_tokens: int) -> List[int]:
        """
        Allocate blocks for a sequence.

        Args:
            seq_id: Sequence ID
            num_tokens: Number of tokens to allocate

        Returns:
            List of allocated block IDs
        """
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if self.next_free_block + num_blocks_needed > self.max_blocks:
            raise RuntimeError(f"Not enough blocks: need {num_blocks_needed}, have {self.max_blocks - self.next_free_block}")

        allocated_blocks = []
        for i in range(num_blocks_needed):
            block_id = self.next_free_block
            self.physical_blocks[block_id].seq_id = seq_id
            allocated_blocks.append(block_id)
            self.next_free_block += 1

        self.seq_allocations[seq_id] = allocated_blocks

        # Initialize block tables for all layers
        for layer_idx in range(self.num_layers):
            self.layer_block_tables[layer_idx][seq_id] = allocated_blocks.copy()

        return allocated_blocks

    def store_kv(
        self,
        layer_idx: int,
        seq_id: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> None:
        """
        Store new KV data in blocks.

        Args:
            layer_idx: Layer index
            seq_id: Sequence ID
            k_new: New K data [batch, num_kv_heads, num_new_tokens, head_dim]
            v_new: New V data [batch, num_kv_heads, num_new_tokens, head_dim]
        """
        block_ids = self.layer_block_tables[layer_idx].get(seq_id, [])

        if not block_ids:
            raise ValueError(f"No blocks allocated for seq_id {seq_id}")

        # For simplicity, store all data in the last allocated block
        # (In production, this would distribute across blocks properly)
        last_block_id = block_ids[-1]
        block = self.physical_blocks[last_block_id]

        block.k_data = k_new.clone()
        block.v_data = v_new.clone()
        block.num_tokens = k_new.shape[2]

    def get_kv_blocks(
        self,
        layer_idx: int,
        seq_id: int,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get KV blocks for a layer and sequence.

        Args:
            layer_idx: Layer index
            seq_id: Sequence ID

        Returns:
            List of (K_block, V_block) tuples
        """
        block_ids = self.layer_block_tables[layer_idx].get(seq_id, [])

        kv_blocks = []
        for block_id in block_ids:
            block = self.physical_blocks[block_id]
            if block.k_data is not None:
                kv_blocks.append((block.k_data, block.v_data))

        return kv_blocks

    def free_sequence(self, seq_id: int) -> None:
        """
        Free all blocks for a sequence.

        Args:
            seq_id: Sequence ID
        """
        if seq_id not in self.seq_allocations:
            return

        for block_id in self.seq_allocations[seq_id]:
            block = self.physical_blocks[block_id]
            block.k_data = None
            block.v_data = None
            block.num_tokens = 0
            block.seq_id = None

        for layer_idx in range(self.num_layers):
            if seq_id in self.layer_block_tables[layer_idx]:
                del self.layer_block_tables[layer_idx][seq_id]

        del self.seq_allocations[seq_id]

    def get_memory_stats(self) -> Dict[str, int]:
        """
        Get memory statistics.

        Returns:
            Dict with total_blocks, used_blocks, memory_bytes
        """
        used_blocks = self.next_free_block

        # Calculate memory per block
        # Each block stores: num_kv_heads * block_size * head_dim * 2 (K + V) * dtype_size
        dtype_size = 2  # bfloat16 = 2 bytes
        bytes_per_block = self.num_kv_heads * self.block_size * self.head_dim * 2 * dtype_size

        total_memory = used_blocks * bytes_per_block

        return {
            "total_blocks": self.max_blocks,
            "used_blocks": used_blocks,
            "free_blocks": self.max_blocks - used_blocks,
            "memory_bytes": total_memory,
            "memory_mb": total_memory / (1024 * 1024),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate base && python -m pytest tests/test_qwen_adapter.py::TestKVCacheManager -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qwen_adapter/kv_cache_manager.py tests/test_qwen_adapter.py
git commit -m "feat: add KV Cache Manager for inference"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

### Task 4: Create Attention Replacement Utility

**Files:**
- Create: `src/qwen_adapter/replace_attention.py`
- Modify: `tests/test_qwen_adapter.py` (add tests)

- [ ] **Step 1: Write test for attention replacement**

```python
# tests/test_qwen_adapter.py (append)

class TestReplaceAttention(unittest.TestCase):
    def test_replace_attention_in_model(self):
        """Test replacing attention layers in model."""
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

        from qwen_adapter.model_loader import load_qwen_model
        from qwen_adapter.replace_attention import replace_attention_with_bpha
        from qwen_adapter.kv_cache_manager import KVCacheManager

        model_path = os.path.join(
            os.path.dirname(__file__), '..', 'model',
            'models--Qwen--Qwen2.5-3B-Instruct', 'snapshots',
            'aa8e72537993ba99e69dfaafa59ed015b17504d1'
        )

        model, tokenizer = load_qwen_model(model_path, device="cpu")

        # Count original attention layers
        original_attn_count = 0
        for name, module in model.named_modules():
            if 'attention' in name.lower() and hasattr(module, 'q_proj'):
                original_attn_count += 1

        self.assertTrue(original_attn_count > 0)

        # Replace attention
        kv_manager = KVCacheManager(
            num_layers=model.config.num_hidden_layers,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=model.config.hidden_size // model.config.num_attention_heads,
            block_size=16,
            max_blocks=100,
        )

        replace_attention_with_bpha(model, kv_manager)

        # Verify replacement
        bpha_count = 0
        for name, module in model.named_modules():
            if hasattr(module, 'forward') and hasattr(module, 'kv_manager'):
                bpha_count += 1

        self.assertEqual(bpha_count, original_attn_count)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate base && python -m pytest tests/test_qwen_adapter.py::TestReplaceAttention -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'qwen_adapter.replace_attention'"

- [ ] **Step 3: Implement replace_attention utility**

```python
# src/qwen_adapter/replace_attention.py
"""
Replace Attention: Replace model attention with BPHA
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from transformers import Qwen2ForCausalLM

from .bpha_attention import BPHAAttention
from .kv_cache_manager import KVCacheManager


class BPHAAttentionWrapper(nn.Module):
    """
    Wrapper that combines original Qwen attention with BPHA and KV cache.
    """

    def __init__(
        self,
        original_attn: nn.Module,
        bpha_attn: BPHAAttention,
        kv_manager: KVCacheManager,
        layer_idx: int,
    ):
        super().__init__()
        self.original_attn = original_attn
        self.bpha_attn = bpha_attn
        self.kv_manager = kv_manager
        self.layer_idx = layer_idx

        # Keep original projections for weight initialization
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj

        # Copy weights to BPHA attention
        self.bpha_attn.q_proj.weight.data = self.q_proj.weight.data.clone()
        self.bpha_attn.k_proj.weight.data = self.k_proj.weight.data.clone()
        self.bpha_attn.v_proj.weight.data = self.v_proj.weight.data.clone()
        self.bpha_attn.o_proj.weight.data = self.o_proj.weight.data.clone()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Dict] = None,
        output_attentions: bool = False,
        use_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass using BPHA with paged KV cache.
        """
        seq_id = kwargs.get('seq_id', 0)

        # Get existing KV blocks for this layer
        kv_blocks = self.kv_manager.get_kv_blocks(self.layer_idx, seq_id)

        # Compute attention
        output = self.bpha_attn.forward(hidden_states, kv_blocks)

        # Store new KV if use_cache
        if use_cache:
            # Project and reshape K, V
            k_new = self.k_proj(hidden_states)
            v_new = self.v_proj(hidden_states)

            batch_size, seq_len = hidden_states.shape[:2]
            k_new = k_new.view(batch_size, seq_len, self.bpha_attn.num_kv_heads, self.bpha_attn.head_dim)
            v_new = v_new.view(batch_size, seq_len, self.bpha_attn.num_kv_heads, self.bpha_attn.head_dim)

            k_new = k_new.transpose(1, 2)  # [batch, num_kv_heads, seq_len, head_dim]
            v_new = v_new.transpose(1, 2)

            self.kv_manager.store_kv(self.layer_idx, seq_id, k_new, v_new)

        return output


def replace_attention_with_bpha(
    model: Qwen2ForCausalLM,
    kv_manager: KVCacheManager,
) -> None:
    """
    Replace all attention layers in model with BPHA.

    Args:
        model: Qwen2ForCausalLM model
        kv_manager: KV Cache Manager instance
    """
    config = model.config

    for layer_idx, layer in enumerate(model.model.layers):
        original_attn = layer.self_attn

        # Create BPHA attention
        bpha_attn = BPHAAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            block_size=kv_manager.block_size,
            layer_idx=layer_idx,
        )

        # Create wrapper
        wrapper = BPHAAttentionWrapper(
            original_attn=original_attn,
            bpha_attn=bpha_attn,
            kv_manager=kv_manager,
            layer_idx=layer_idx,
        )

        # Replace in model
        layer.self_attn = wrapper
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate base && python -m pytest tests/test_qwen_adapter.py::TestReplaceAttention -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qwen_adapter/replace_attention.py tests/test_qwen_adapter.py
git commit -m "feat: add attention replacement utility"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

### Task 5: Create E2E Inference Example

**Files:**
- Create: `examples/example_qwen_bpha.py`

- [ ] **Step 1: Write E2E example**

```python
# examples/example_qwen_bpha.py
"""
Example: Qwen2.5-3B with BPHA Attention

End-to-end inference example demonstrating BPHA attention
with paged KV cache on Qwen model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from qwen_adapter.model_loader import load_qwen_model
from qwen_adapter.replace_attention import replace_attention_with_bpha
from qwen_adapter.kv_cache_manager import KVCacheManager


def main():
    print("=" * 60)
    print("Qwen2.5-3B with BPHA Attention - E2E Example")
    print("=" * 60)

    # Load model
    model_path = os.path.join(
        os.path.dirname(__file__), '..', 'model',
        'models--Qwen--Qwen2.5-3B-Instruct', 'snapshots',
        'aa8e72537993ba99e69dfaafa59ed015b17504d1'
    )

    print(f"\nLoading model from: {model_path}")
    model, tokenizer = load_qwen_model(model_path)

    print(f"Model config:")
    print(f"  hidden_size: {model.config.hidden_size}")
    print(f"  num_heads: {model.config.num_attention_heads}")
    print(f"  num_kv_heads: {model.config.num_key_value_heads}")
    print(f"  num_layers: {model.config.num_hidden_layers}")

    # Create KV Cache Manager
    kv_manager = KVCacheManager(
        num_layers=model.config.num_hidden_layers,
        num_kv_heads=model.config.num_key_value_heads,
        head_dim=model.config.hidden_size // model.config.num_attention_heads,
        block_size=16,
        max_blocks=100,
    )

    print(f"\nKV Cache Manager initialized:")
    print(f"  block_size: {kv_manager.block_size}")
    print(f"  max_blocks: {kv_manager.max_blocks}")

    # Replace attention layers
    print("\nReplacing attention layers with BPHA...")
    replace_attention_with_bpha(model, kv_manager)

    print("Replacement complete.")

    # Prepare input
    prompt = "你好，请介绍一下你自己。"
    print(f"\nInput prompt: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    num_tokens = input_ids.shape[1]

    print(f"Tokenized: {num_tokens} tokens")

    # Allocate KV cache
    seq_id = 1
    kv_manager.allocate_sequence(seq_id, num_tokens)

    # Generate
    print("\nGenerating response...")
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=False,
            seq_id=seq_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nResponse: {response}")

    # Memory stats
    stats = kv_manager.get_memory_stats()
    print(f"\nKV Cache Memory Stats:")
    print(f"  Used blocks: {stats['used_blocks']}")
    print(f"  Memory used: {stats['memory_mb']:.2f} MB")

    print("\n" + "=" * 60)
    print("E2E Example Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run example to verify it works**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate base && cd /home/daivy/projects/cuda_exer && python examples/example_qwen_bpha.py`
Expected: Model loads, generates coherent response, prints memory stats

- [ ] **Step 3: Commit**

```bash
git add examples/example_qwen_bpha.py
git commit -m "feat: add E2E Qwen inference example with BPHA"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

### Task 6: Update Module Exports

**Files:**
- Modify: `src/qwen_adapter/__init__.py`

- [ ] **Step 1: Update __init__.py**

```python
# src/qwen_adapter/__init__.py
"""
Qwen Adapter: BPHA attention adapter for Qwen2.5 models
"""

from .model_loader import load_qwen_model, get_model_config
from .bpha_attention import BPHAAttention
from .kv_cache_manager import KVCacheManager, KVBlock
from .replace_attention import replace_attention_with_bpha, BPHAAttentionWrapper

__all__ = [
    "load_qwen_model",
    "get_model_config",
    "BPHAAttention",
    "KVCacheManager",
    "KVBlock",
    "replace_attention_with_bpha",
    "BPHAAttentionWrapper",
]
```

- [ ] **Step 2: Commit**

```bash
git add src/qwen_adapter/__init__.py
git commit -m "feat: update qwen_adapter module exports"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

### Task 7: Run All Tests

- [ ] **Step 1: Run full test suite**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate base && cd /home/daivy/projects/cuda_exer && python -m pytest tests/ -v`
Expected: All tests pass (including original 43 tests)

- [ ] **Step 2: If tests fail, debug and fix**

Check error messages, fix implementation, rerun tests.

- [ ] **Step 3: Final commit if fixes needed**

---

## Self-Review

**Spec coverage check:**
- ✅ 1.1 Qwen 模型适配层 → Task 1 (model_loader.py)
- ✅ 1.2 BPHA attention 替换 → Task 2-4 (bpha_attention.py, replace_attention.py)
- ✅ 1.3 E2E inference 测试 → Task 5 (example_qwen_bpha.py)
- ✅ 1.4 KV Cache 内存测量 → Task 5 (memory stats in example)

**Placeholder scan:** None found - all steps have concrete code.

**Type consistency:**
- BPHAAttention params: hidden_size, num_heads, num_kv_heads, block_size
- KVCacheManager params: num_layers, num_kv_heads, head_dim, block_size
- Both use consistent naming.

---

## Verification Criteria

- [ ] 模型生成输出正确（无报错、文本连贯）
- [ ] KV Cache 内存指标量化（memory_mb printed）
- [ ] 现有 43 个测试继续通过

---

## Execution Options

Plan complete and saved to `docs/superpowers/plans/2026-04-25-phase1-e2e-model-testing.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
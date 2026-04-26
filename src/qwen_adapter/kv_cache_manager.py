"""KV Cache Manager for multi-layer transformer inference.

Manages paged KV cache for Qwen2.5-3B (36 layers) during inference.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class KVBlock:
    """Represents a physical KV cache block."""

    block_id: int
    k_data: Optional[torch.Tensor] = None
    v_data: Optional[torch.Tensor] = None
    num_tokens: int = 0
    seq_id: Optional[int] = None


class KVCacheManager:
    """Manages paged KV cache for multi-layer transformer inference.

    This class handles:
    - Physical block pool management
    - Per-layer block tables (logical-to-physical mapping)
    - Sequence allocation and deallocation
    - KV cache storage and retrieval

    Args:
        num_layers: Number of transformer layers (e.g., 36 for Qwen2.5-3B)
        num_kv_heads: Number of KV heads (for GQA)
        head_dim: Dimension of each attention head
        block_size: Number of tokens per block
        max_blocks: Maximum number of physical blocks in the pool
        device: Device for tensor storage (default: "cuda")
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        max_blocks: int,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device

        # Physical block pool: block_id -> KVBlock
        self.physical_blocks: Dict[int, KVBlock] = {}
        self.free_block_ids: List[int] = list(range(max_blocks))

        # Per-layer block tables: layer_idx -> {seq_id -> [block_ids]}
        self.layer_block_tables: List[Dict[int, List[int]]] = [
            {} for _ in range(num_layers)
        ]

        # Sequence allocations: seq_id -> list of allocated block_ids
        self.seq_allocations: Dict[int, List[int]] = {}

        # Sequence token counts: seq_id -> current number of tokens allocated
        self.seq_token_counts: Dict[int, int] = {}

        # Stored token counts: seq_id -> actual number of tokens stored in cache
        self.seq_stored_tokens: Dict[int, int] = {}

    def allocate_sequence(self, seq_id: int, num_tokens: int) -> List[int]:
        """Allocate blocks for a sequence.

        Args:
            seq_id: Unique sequence identifier
            num_tokens: Number of tokens to allocate space for

        Returns:
            List of allocated block IDs

        Raises:
            RuntimeError: If not enough free blocks available
        """
        if seq_id in self.seq_allocations:
            raise RuntimeError(f"Sequence {seq_id} already allocated")

        # Calculate number of blocks needed
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if num_blocks_needed > len(self.free_block_ids):
            raise RuntimeError(
                f"Not enough free blocks: need {num_blocks_needed}, "
                f"have {len(self.free_block_ids)}"
            )

        # Allocate blocks
        allocated_blocks = []
        for _ in range(num_blocks_needed):
            block_id = self.free_block_ids.pop(0)

            # Create KVBlock for each layer
            for layer_idx in range(self.num_layers):
                block = KVBlock(
                    block_id=block_id,
                    k_data=torch.zeros(
                        self.batch_size, self.num_kv_heads, self.block_size, self.head_dim,
                        device=self.device,
                        dtype=self.dtype,
                    ),
                    v_data=torch.zeros(
                        self.batch_size, self.num_kv_heads, self.block_size, self.head_dim,
                        device=self.device,
                        dtype=self.dtype,
                    ),
                    num_tokens=0,
                    seq_id=seq_id,
                )
                # Store in physical_blocks with layer-specific key
                self.physical_blocks[(layer_idx, block_id)] = block

                # Add to layer block table
                if seq_id not in self.layer_block_tables[layer_idx]:
                    self.layer_block_tables[layer_idx][seq_id] = []
                self.layer_block_tables[layer_idx][seq_id].append(block_id)

            allocated_blocks.append(block_id)

        self.seq_allocations[seq_id] = allocated_blocks
        self.seq_token_counts[seq_id] = num_tokens
        self.seq_stored_tokens[seq_id] = 0  # Initialize as 0, will be updated by store_kv

        return allocated_blocks

    def store_kv(
        self,
        layer_idx: int,
        seq_id: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> None:
        """Store new KV cache data for a layer.

        Args:
            layer_idx: Layer index (0 to num_layers-1)
            seq_id: Sequence identifier
            k_new: New K tensor [batch, num_kv_heads, num_tokens, head_dim]
            v_new: New V tensor [batch, num_kv_heads, num_tokens, head_dim]

        Raises:
            RuntimeError: If sequence not allocated
            ValueError: If tensor shapes don't match
        """
        if seq_id not in self.seq_allocations:
            raise RuntimeError(f"Sequence {seq_id} not allocated")

        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(
                f"Invalid layer_idx {layer_idx}, "
                f"must be 0 to {self.num_layers - 1}"
            )

        # Get block IDs for this sequence and layer
        block_ids = self.layer_block_tables[layer_idx].get(seq_id, [])

        if not block_ids:
            raise RuntimeError(
                f"No blocks allocated for seq_id={seq_id}, layer_idx={layer_idx}"
            )

        # Store data in blocks
        num_tokens = k_new.shape[2]
        tokens_stored = 0

        for block_id in block_ids:
            block = self.physical_blocks[(layer_idx, block_id)]
            block_tokens = min(
                self.block_size - block.num_tokens,
                num_tokens - tokens_stored
            )

            if block_tokens <= 0:
                continue

            # Copy tokens to block
            start_idx = block.num_tokens
            end_idx = start_idx + block_tokens

            # block.k_data: [batch, num_kv_heads, block_size, head_dim]
            # k_new[0, :, tokens_stored:tokens_stored + block_tokens, :]: [num_kv_heads, block_tokens, head_dim]
            block.k_data[:, :, start_idx:end_idx, :] = k_new[0, :, tokens_stored:tokens_stored + block_tokens, :]
            block.v_data[:, :, start_idx:end_idx, :] = v_new[0, :, tokens_stored:tokens_stored + block_tokens, :]

            block.num_tokens = end_idx
            tokens_stored += block_tokens

            if tokens_stored >= num_tokens:
                break

        # Note: We don't update seq_stored_tokens here because it would be
        # incremented multiple times (once per layer). Instead, we track
        # stored tokens using block.num_tokens and update seq_stored_tokens
        # externally when needed.

    def get_kv_blocks(
        self,
        layer_idx: int,
        seq_id: int,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get KV cache blocks for a layer and sequence.

        Args:
            layer_idx: Layer index
            seq_id: Sequence identifier

        Returns:
            List of (K, V) tensor tuples for each block
        """
        if seq_id not in self.seq_allocations:
            raise RuntimeError(f"Sequence {seq_id} not allocated")

        block_ids = self.layer_block_tables[layer_idx].get(seq_id, [])

        kv_blocks = []
        for block_id in block_ids:
            block = self.physical_blocks[(layer_idx, block_id)]
            if block.num_tokens > 0:
                # Only return valid tokens, not the full block_size
                k_valid = block.k_data[:, :, :block.num_tokens, :]
                v_valid = block.v_data[:, :, :block.num_tokens, :]
                kv_blocks.append((k_valid, v_valid))

        return kv_blocks

    def extend_sequence(self, seq_id: int, additional_tokens: int) -> None:
        """Extend a sequence to accommodate more tokens.

        Allocates additional blocks if needed for the new tokens.

        Args:
            seq_id: Sequence identifier
            additional_tokens: Number of additional tokens to allocate space for

        Raises:
            RuntimeError: If sequence not allocated or not enough free blocks
        """
        if seq_id not in self.seq_allocations:
            raise RuntimeError(f"Sequence {seq_id} not allocated")

        current_tokens = self.seq_token_counts[seq_id]
        total_tokens = current_tokens + additional_tokens

        # Calculate current and needed blocks
        current_blocks = len(self.seq_allocations[seq_id])
        needed_blocks = (total_tokens + self.block_size - 1) // self.block_size
        additional_blocks = needed_blocks - current_blocks

        if additional_blocks <= 0:
            # Already have enough blocks
            self.seq_token_counts[seq_id] = total_tokens
            return

        if additional_blocks > len(self.free_block_ids):
            raise RuntimeError(
                f"Not enough free blocks: need {additional_blocks}, "
                f"have {len(self.free_block_ids)}"
            )

        # Allocate additional blocks
        for _ in range(additional_blocks):
            block_id = self.free_block_ids.pop(0)

            # Create KVBlock for each layer
            for layer_idx in range(self.num_layers):
                block = KVBlock(
                    block_id=block_id,
                    k_data=torch.zeros(
                        self.batch_size, self.num_kv_heads, self.block_size, self.head_dim,
                        device=self.device,
                        dtype=self.dtype,
                    ),
                    v_data=torch.zeros(
                        self.batch_size, self.num_kv_heads, self.block_size, self.head_dim,
                        device=self.device,
                        dtype=self.dtype,
                    ),
                    num_tokens=0,
                    seq_id=seq_id,
                )
                self.physical_blocks[(layer_idx, block_id)] = block
                self.layer_block_tables[layer_idx][seq_id].append(block_id)

            self.seq_allocations[seq_id].append(block_id)

        self.seq_token_counts[seq_id] = total_tokens

    def get_sequence_length(self, seq_id: int) -> int:
        """Get the current number of tokens stored for a sequence.

        Args:
            seq_id: Sequence identifier

        Returns:
            Number of tokens actually stored in the sequence

        Raises:
            RuntimeError: If sequence not allocated
        """
        if seq_id not in self.seq_allocations:
            raise RuntimeError(f"Sequence {seq_id} not allocated")

        # Calculate stored tokens from block.num_tokens (for any layer)
        # All layers should have the same number of stored tokens
        block_ids = self.seq_allocations[seq_id]
        if not block_ids:
            return 0

        # Use layer 0's blocks to calculate stored tokens
        total_tokens = 0
        for block_id in block_ids:
            key = (0, block_id)  # Use layer 0
            if key in self.physical_blocks:
                total_tokens += self.physical_blocks[key].num_tokens

        return total_tokens

    def get_allocated_length(self, seq_id: int) -> int:
        """Get the allocated token capacity for a sequence.

        Args:
            seq_id: Sequence identifier

        Returns:
            Maximum number of tokens that can be stored

        Raises:
            RuntimeError: If sequence not allocated
        """
        if seq_id not in self.seq_token_counts:
            raise RuntimeError(f"Sequence {seq_id} not allocated")
        return self.seq_token_counts[seq_id]

    def free_sequence(self, seq_id: int) -> None:
        """Free all blocks allocated for a sequence.

        Args:
            seq_id: Sequence identifier to free
        """
        if seq_id not in self.seq_allocations:
            return

        block_ids = self.seq_allocations[seq_id]

        # Free physical blocks for all layers
        for block_id in block_ids:
            for layer_idx in range(self.num_layers):
                key = (layer_idx, block_id)
                if key in self.physical_blocks:
                    del self.physical_blocks[key]

                if seq_id in self.layer_block_tables[layer_idx]:
                    self.layer_block_tables[layer_idx][seq_id] = []

            # Return block to free pool
            self.free_block_ids.append(block_id)

        # Remove sequence from tracking
        del self.seq_allocations[seq_id]
        if seq_id in self.seq_token_counts:
            del self.seq_token_counts[seq_id]
        if seq_id in self.seq_stored_tokens:
            del self.seq_stored_tokens[seq_id]

    def get_memory_stats(self) -> Dict:
        """Get memory statistics for the KV cache.

        Returns:
            Dict with memory statistics:
            - total_blocks: Total number of blocks in pool
            - used_blocks: Number of blocks currently in use
            - free_blocks: Number of free blocks
            - memory_bytes: Total memory used by KV cache (in bytes)
        """
        used_blocks = self.max_blocks - len(self.free_block_ids)

        # Calculate memory per block
        # K and V each: num_kv_heads * block_size * head_dim * 4 bytes (float32)
        bytes_per_element = 4  # float32
        block_memory = (
            2  # K and V
            * self.num_kv_heads
            * self.block_size
            * self.head_dim
            * bytes_per_element
        )

        # Total memory for all layers
        total_memory = used_blocks * block_memory * self.num_layers

        return {
            "total_blocks": self.max_blocks,
            "used_blocks": used_blocks,
            "free_blocks": len(self.free_block_ids),
            "memory_bytes": total_memory,
            "memory_mb": total_memory / (1024 * 1024),
        }
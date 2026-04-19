"""
Example: Blocked Tensor

Demonstrates compiler-friendly blocked tensor operations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from blockedTensor import BlockedTensor, TensorLayout, LayoutConstraint


def main():
    print("=" * 60)
    print("Blocked Tensor Example")
    print("=" * 60)

    print("\n--- Creating BlockedTensor ---")
    base_shape = (1024, 128)
    block_size = (16, 16)

    bt = BlockedTensor(base_shape=base_shape, block_size=block_size)
    print(f"Created: {bt}")

    print(f"\nLayout info:")
    layout_info = bt.get_layout_info()
    for key, value in layout_info.items():
        print(f"  {key}: {value}")

    print("\n--- Setting and Getting Blocks ---")
    block_idx = 5
    mock_data = np.random.randn(*block_size).astype(np.float32)

    bt.set_block(block_idx, mock_data)
    print(f"Set block {block_idx} with data shape {mock_data.shape}")

    retrieved = bt.get_block(block_idx)
    print(f"Retrieved block shape: {retrieved.shape}")

    diff = np.max(np.abs(mock_data - retrieved))
    print(f"Max difference: {diff:.2e}")

    if diff < 1e-6:
        print("✓ Block data integrity verified!")
    else:
        print("✗ Warning: Data mismatch")

    print("\n--- TensorLayout Demo ---")
    mock_tensor = np.zeros((512, 64), dtype=np.float32)
    mock_tensor[::2, ::2] = 1.0

    layout = TensorLayout.from_tensor(mock_tensor, block_size=(16, 16))
    print(f"Created layout: {layout}")

    print(f"\nLayout properties:")
    print(f"  Is contiguous: {layout.is_contiguous()}")
    print(f"  Is blocked: {layout.is_blocked()}")
    print(f"  Cache friendliness: {layout.estimate_cache_friendliness():.2f}")

    suggested_tile = layout.suggest_tile_shape(cache_size=16384, element_size=4)
    print(f"  Suggested tile shape (16KB cache): {suggested_tile}")

    print("\n--- Layout Constraints ---")
    constraint = LayoutConstraint(
        alignment=32,
        contiguity='blocked-random',
        access_pattern='blocked-random',
        prefer_blocked=True
    )
    print(f"Constraint: alignment={constraint.alignment}, pattern={constraint.access_pattern}")


if __name__ == "__main__":
    main()
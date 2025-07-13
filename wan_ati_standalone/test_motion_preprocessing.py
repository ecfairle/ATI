#!/usr/bin/env python3
"""Test script to debug motion preprocessing"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wan.preprocessing import MotionPreprocessor

# Test motion preprocessing
print("Testing motion preprocessing...")

# Create test tracks tensor with shape (T, N, 4)
# T=81 frames, N=10 tracks, 4=[batch_idx, x, y, visibility]
tracks = torch.randn(81, 10, 4)
print(f"Input tracks shape: {tracks.shape}")

# Test preprocessing
preprocessor = MotionPreprocessor()
normalized = preprocessor.normalize_tracks(
    tracks,
    width=1280,
    height=720,
    device=torch.device('cpu')
)

print(f"Output tracks shape: {normalized.shape}")
print(f"Expected shape for patch_motion: (B=1, T=81, N=10, 4)")

# Test the normalization that happens in patch_motion
if normalized.shape == (1, 81, 10, 4):
    print("\n✓ Shape is correct for patch_motion!")
    
    # Simulate what happens in patch_motion
    _, T, H, W = 16, 21, 90, 160  # Example video shape
    N = normalized.shape[2]
    
    # Split tracks like in patch_motion
    _, tracks_xy, visible = torch.split(normalized, [1, 2, 1], dim=-1)
    print(f"\nAfter split:")
    print(f"  tracks_xy shape: {tracks_xy.shape}")
    print(f"  visible shape: {visible.shape}")
    
    # Test normalization
    min_dim = min(H, W)
    normalization_factor = torch.tensor([W / min_dim, H / min_dim])
    print(f"\nNormalization factor shape: {normalization_factor.shape}")
    print(f"Normalization factor: {normalization_factor}")
    
    # This is where the error might occur
    try:
        tracks_n = tracks_xy / normalization_factor
        print(f"✓ Normalization successful! Result shape: {tracks_n.shape}")
    except RuntimeError as e:
        print(f"✗ Error during normalization: {e}")
        print(f"  tracks_xy shape: {tracks_xy.shape}")
        print(f"  normalization_factor shape: {normalization_factor.shape}")
else:
    print(f"\n✗ Shape mismatch! Got {normalized.shape}, expected (1, 81, 10, 4)")
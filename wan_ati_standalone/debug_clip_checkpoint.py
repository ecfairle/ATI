#!/usr/bin/env python3
"""Debug CLIP checkpoint to understand its structure."""

import sys
from safetensors.torch import load_file

if len(sys.argv) < 2:
    print("Usage: python debug_clip_checkpoint.py /path/to/clip_vision_h.safetensors")
    sys.exit(1)

checkpoint_path = sys.argv[1]
print(f"Loading checkpoint: {checkpoint_path}\n")

# Load the state dict
state_dict = load_file(checkpoint_path)

print(f"Total keys in checkpoint: {len(state_dict)}\n")

# Group keys by prefix
prefixes = {}
for key in sorted(state_dict.keys()):
    prefix = key.split('.')[0]
    if prefix not in prefixes:
        prefixes[prefix] = []
    prefixes[prefix].append(key)

print("Key prefixes found:")
for prefix, keys in prefixes.items():
    print(f"  {prefix}: {len(keys)} keys")

print("\nFirst 20 keys:")
for i, key in enumerate(sorted(state_dict.keys())[:20]):
    shape = state_dict[key].shape
    print(f"  {key}: {shape}")

print("\nLast 10 keys:")
for key in sorted(state_dict.keys())[-10:]:
    shape = state_dict[key].shape
    print(f"  {key}: {shape}")

# Check for specific expected keys
print("\nChecking for expected VisionTransformer keys:")
expected_patterns = [
    "patch_embedding",
    "cls_embedding", 
    "pos_embedding",
    "transformer",
    "head",
    "pre_norm",
    "post_norm"
]

for pattern in expected_patterns:
    found = [k for k in state_dict.keys() if pattern in k]
    if found:
        print(f"  ✓ Found {pattern}: {len(found)} keys")
        print(f"    Example: {found[0]}")
    else:
        print(f"  ✗ Missing {pattern}")

# Try to identify the model structure
print("\nTrying to identify model structure...")
if any("visual." in k for k in state_dict.keys()):
    print("  - Keys contain 'visual.' prefix (full CLIP model)")
elif any("vision_model." in k for k in state_dict.keys()):
    print("  - Keys contain 'vision_model.' prefix (HuggingFace format)")
elif any("model." in k for k in state_dict.keys()):
    print("  - Keys contain 'model.' prefix")
else:
    print("  - No common prefix found, likely direct vision transformer")
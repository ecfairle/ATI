#!/usr/bin/env python3
"""
Check the actual dtype of tensors in the checkpoint
"""

import sys
from safetensors.torch import load_file
import torch

def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/Wan2_1-I2V-ATI-14B_fp8_e4m3fn.safetensors"
    
    print(f"Checking checkpoint: {checkpoint_path}")
    
    # Load just the metadata without loading tensors to memory
    with open(checkpoint_path, 'rb') as f:
        # safetensors format stores metadata at the beginning
        header_size = int.from_bytes(f.read(8), 'little')
        header = f.read(header_size).decode('utf-8')
        
    print(f"\nHeader size: {header_size} bytes")
    
    # Parse the header (it's JSON)
    import json
    metadata = json.loads(header)
    
    # Check dtypes
    dtypes = {}
    total_params = 0
    
    for tensor_name, tensor_info in metadata.items():
        if isinstance(tensor_info, dict) and 'dtype' in tensor_info:
            dtype = tensor_info['dtype']
            shape = tensor_info.get('shape', [])
            numel = 1
            for dim in shape:
                numel *= dim
            
            if dtype not in dtypes:
                dtypes[dtype] = {'count': 0, 'params': 0}
            
            dtypes[dtype]['count'] += 1
            dtypes[dtype]['params'] += numel
            total_params += numel
            
            # Show first few tensors
            if dtypes[dtype]['count'] <= 3:
                print(f"  {tensor_name}: dtype={dtype}, shape={shape}")
    
    print(f"\nTotal parameters: {total_params / 1e9:.2f}B")
    print("\nDtype distribution:")
    for dtype, info in dtypes.items():
        print(f"  {dtype}: {info['count']} tensors, {info['params'] / 1e9:.2f}B parameters ({info['params'] / total_params * 100:.1f}%)")
    
    # Try to load a single tensor to check actual dtype
    print("\nLoading a sample tensor to verify dtype...")
    try:
        state_dict = load_file(checkpoint_path, device='cpu')
        first_key = next(iter(state_dict.keys()))
        first_tensor = state_dict[first_key]
        print(f"  {first_key}: loaded dtype={first_tensor.dtype}, shape={first_tensor.shape}")
        
        # Check if it's actually fp8
        if hasattr(torch, 'float8_e4m3fn'):
            print(f"  torch.float8_e4m3fn available: True")
            if first_tensor.dtype == torch.float8_e4m3fn:
                print("  Tensor is in FP8 format!")
            else:
                print(f"  Tensor is NOT in FP8 format, it's {first_tensor.dtype}")
        else:
            print("  torch.float8_e4m3fn NOT available in this PyTorch version")
            
    except Exception as e:
        print(f"  Error loading tensor: {e}")

if __name__ == "__main__":
    main()
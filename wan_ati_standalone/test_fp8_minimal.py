#!/usr/bin/env python3
"""
Minimal test to check FP8 loading
"""

import torch
import logging
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)

def main():
    # Check FP8 support
    fp8_supported = hasattr(torch, 'float8_e4m3fn')
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"FP8 support: {fp8_supported}")
    
    if not fp8_supported:
        logging.error("FP8 not supported in this PyTorch version!")
        return
    
    # Test simple FP8 tensor
    logging.info("\nTesting FP8 tensor creation:")
    x = torch.randn(1000, 1000, dtype=torch.float32)
    logging.info(f"Float32 tensor: {x.dtype}, size: {x.numel() * 4 / 1024**2:.2f} MB")
    
    x_fp8 = x.to(torch.float8_e4m3fn)
    logging.info(f"FP8 tensor: {x_fp8.dtype}, size: {x_fp8.numel() * 1 / 1024**2:.2f} MB")
    
    # Test moving to GPU
    if torch.cuda.is_available():
        logging.info("\nTesting GPU transfer:")
        x_fp8_gpu = x_fp8.cuda()
        logging.info(f"FP8 on GPU: {x_fp8_gpu.dtype}, device: {x_fp8_gpu.device}")
        
        # Check memory usage
        allocated = torch.cuda.memory_allocated() / 1024**3
        logging.info(f"GPU memory allocated: {allocated:.3f} GB")
    
    # Now test loading a model checkpoint
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else "/workspace"
    checkpoint_path = os.path.join(checkpoint_dir, 'Wan2_1-I2V-ATI-14B_fp8_e4m3fn.safetensors')
    
    if os.path.exists(checkpoint_path):
        logging.info(f"\nTesting checkpoint loading from {checkpoint_path}")
        
        from wan.utils.fp8_utils import load_fp8_checkpoint
        
        # Try loading with FP8 preservation
        state_dict = load_fp8_checkpoint(
            checkpoint_path,
            device='cpu',
            keep_fp8=True
        )
        
        # Check dtypes
        dtype_counts = {}
        for name, tensor in list(state_dict.items())[:10]:  # Check first 10
            dtype = str(tensor.dtype)
            if dtype not in dtype_counts:
                dtype_counts[dtype] = 0
            dtype_counts[dtype] += 1
            logging.info(f"  {name}: {tensor.dtype}, shape: {tensor.shape}")
        
        logging.info(f"\nDtype summary: {dtype_counts}")

if __name__ == "__main__":
    main()
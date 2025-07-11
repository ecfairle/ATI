#!/usr/bin/env python3
"""
Test script to check memory usage and FP8 support
"""

import torch
import logging
import os
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)]
)

def test_fp8_support():
    """Check if PyTorch supports FP8 operations"""
    logging.info("Testing FP8 support...")
    
    # Check for FP8 dtypes
    fp8_e4m3fn = hasattr(torch, 'float8_e4m3fn')
    fp8_e5m2 = hasattr(torch, 'float8_e5m2')
    
    logging.info(f"torch.float8_e4m3fn available: {fp8_e4m3fn}")
    logging.info(f"torch.float8_e5m2 available: {fp8_e5m2}")
    
    # Check PyTorch version
    logging.info(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA version
    if torch.cuda.is_available():
        logging.info(f"CUDA available: True")
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"CUDNN version: {torch.backends.cudnn.version()}")
        
        # Check GPU info
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logging.info(f"GPU {i}: {props.name}")
            logging.info(f"  Compute capability: {props.major}.{props.minor}")
            logging.info(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
            
            # Check current memory usage
            if torch.cuda.is_initialized():
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                logging.info(f"  Memory allocated: {allocated:.2f} GB")
                logging.info(f"  Memory reserved: {reserved:.2f} GB")
    else:
        logging.warning("CUDA not available")
    
    # Test creating tensors with different dtypes
    logging.info("\nTesting tensor creation with different dtypes...")
    test_dtypes = ['float32', 'float16', 'bfloat16']
    
    if fp8_e4m3fn:
        test_dtypes.append('float8_e4m3fn')
    if fp8_e5m2:
        test_dtypes.append('float8_e5m2')
    
    for dtype_name in test_dtypes:
        try:
            dtype = getattr(torch, dtype_name)
            x = torch.randn(10, 10, dtype=dtype, device='cuda' if torch.cuda.is_available() else 'cpu')
            logging.info(f"  {dtype_name}: Success (shape={x.shape}, dtype={x.dtype})")
        except Exception as e:
            logging.error(f"  {dtype_name}: Failed - {e}")
    
    # Test memory allocation
    if torch.cuda.is_available():
        logging.info("\nTesting memory allocation...")
        try:
            # Try allocating a large tensor
            size_gb = 1.0  # 1GB test allocation
            elements = int(size_gb * 1024**3 / 2)  # 2 bytes per float16
            test_tensor = torch.randn(elements, dtype=torch.float16, device='cuda')
            logging.info(f"Successfully allocated {size_gb} GB tensor")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"Failed to allocate {size_gb} GB tensor: {e}")

def test_model_loading():
    """Test loading a safetensors file"""
    logging.info("\nTesting safetensors loading...")
    
    try:
        from safetensors.torch import load_file
        logging.info("safetensors import successful")
        
        # Check if checkpoint exists
        checkpoint_path = "/workspace/Wan2_1-I2V-ATI-14B_fp8_e4m3fn.safetensors"
        if os.path.exists(checkpoint_path):
            logging.info(f"Checkpoint found at {checkpoint_path}")
            file_size = os.path.getsize(checkpoint_path) / 1024**3
            logging.info(f"Checkpoint size: {file_size:.2f} GB")
            
            # Try to peek at the checkpoint
            try:
                metadata = load_file(checkpoint_path, device='cpu')
                logging.info(f"Number of tensors in checkpoint: {len(metadata)}")
                
                # Check dtypes in checkpoint
                dtypes = set()
                for k, v in list(metadata.items())[:5]:  # Check first 5 tensors
                    dtypes.add(str(v.dtype))
                    logging.info(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                
                logging.info(f"Unique dtypes in checkpoint: {dtypes}")
                
                # Clean up
                del metadata
                torch.cuda.empty_cache()
                
            except Exception as e:
                logging.error(f"Failed to load checkpoint: {e}")
        else:
            logging.warning(f"Checkpoint not found at {checkpoint_path}")
            
    except ImportError:
        logging.error("safetensors not installed")

def main():
    logging.info("=== WAN ATI Memory and FP8 Test ===")
    
    # Run tests
    test_fp8_support()
    test_model_loading()
    
    logging.info("\n=== Test Complete ===")

if __name__ == "__main__":
    main()
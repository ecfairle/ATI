#!/usr/bin/env python3
"""
Test chunked loading of FP8 model
"""

import os
import sys
import torch
import logging
import gc

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wan.utils.fp8_utils import load_fp8_checkpoint

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)]
)

def log_memory(stage):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logging.info(f"[{stage}] GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    # Also log CPU memory usage
    try:
        import psutil
        process = psutil.Process()
        rss_gb = process.memory_info().rss / 1024**3
        logging.info(f"[{stage}] CPU memory (RSS): {rss_gb:.2f}GB")
    except:
        pass

def main():
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else "/workspace"
    safetensors_path = os.path.join(checkpoint_dir, 'Wan2_1-I2V-ATI-14B_fp8_e4m3fn.safetensors')
    
    if not os.path.exists(safetensors_path):
        logging.error(f"Checkpoint not found at {safetensors_path}")
        return
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    log_memory("Initial")
    
    # Test loading with FP8 preservation
    logging.info("Testing FP8 checkpoint loading with chunking...")
    
    try:
        # Load to GPU directly if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        state_dict = load_fp8_checkpoint(
            safetensors_path,
            device=device,
            keep_fp8=True
        )
        
        log_memory("After loading")
        
        # Check dtypes
        dtype_counts = {}
        total_params = 0
        for name, tensor in state_dict.items():
            dtype = str(tensor.dtype)
            if dtype not in dtype_counts:
                dtype_counts[dtype] = 0
            dtype_counts[dtype] += tensor.numel()
            total_params += tensor.numel()
        
        logging.info(f"Total parameters: {total_params / 1e9:.2f}B")
        for dtype, count in dtype_counts.items():
            logging.info(f"  {dtype}: {count / 1e9:.2f}B parameters")
        
        # Check a few tensors
        for i, (name, tensor) in enumerate(state_dict.items()):
            if i < 5:
                logging.info(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
            else:
                break
        
        # Cleanup
        del state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        log_memory("After cleanup")
        
    except Exception as e:
        logging.error(f"Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
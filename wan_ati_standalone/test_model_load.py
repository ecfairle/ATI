#!/usr/bin/env python3
"""
Minimal test to load the model and check memory usage
"""

import os
import sys
import torch
import logging
import gc

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wan.modules.model import WanModel
from wan.configs.wan_i2v_14B import i2v_14B

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
    else:
        logging.info(f"[{stage}] CUDA not available")

def main():
    logging.info("=== Model Loading Test ===")
    
    # Set memory fraction
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else "/workspace"
    safetensors_path = os.path.join(checkpoint_dir, 'Wan2_1-I2V-ATI-14B_fp8_e4m3fn.safetensors')
    
    if not os.path.exists(safetensors_path):
        logging.error(f"Checkpoint not found at {safetensors_path}")
        return
    
    log_memory("Initial")
    
    # Create model config
    model_config = {
        '_class_name': 'WanModel',
        '_diffusers_version': '0.30.0',
        'model_type': 'i2v',
        'text_len': i2v_14B.text_len,
        'in_dim': 36,
        'dim': i2v_14B.dim,
        'ffn_dim': i2v_14B.ffn_dim,
        'freq_dim': i2v_14B.freq_dim,
        'out_dim': 16,
        'num_heads': i2v_14B.num_heads,
        'num_layers': i2v_14B.num_layers,
        'eps': i2v_14B.eps
    }
    
    try:
        # Load model
        logging.info(f"Loading model from {safetensors_path}")
        model = WanModel.from_single_file(safetensors_path, config=model_config)
        log_memory("After model load")
        
        # Check parameter dtypes
        param_dtypes = {}
        total_params = 0
        for name, param in model.named_parameters():
            dtype = str(param.dtype)
            if dtype not in param_dtypes:
                param_dtypes[dtype] = 0
            param_dtypes[dtype] += param.numel()
            total_params += param.numel()
        
        logging.info(f"Total parameters: {total_params / 1e9:.2f}B")
        for dtype, count in param_dtypes.items():
            logging.info(f"  {dtype}: {count / 1e9:.2f}B parameters")
        
        # Try to move to GPU if available
        if torch.cuda.is_available():
            logging.info("Moving model to GPU...")
            model = model.cuda()
            log_memory("After moving to GPU")
            
            # Try a dummy forward pass with minimal input
            logging.info("Testing forward pass...")
            with torch.no_grad():
                # Create minimal dummy inputs
                x = [torch.randn(16, 1, 8, 8, dtype=torch.bfloat16, device='cuda')]
                t = torch.tensor([500], dtype=torch.long, device='cuda')
                context = [torch.randn(512, 4096, dtype=torch.bfloat16, device='cuda')]
                seq_len = 64
                clip_fea = torch.randn(1, 257, 1280, dtype=torch.bfloat16, device='cuda')
                y = [torch.randn(20, 1, 8, 8, dtype=torch.bfloat16, device='cuda')]
                
                try:
                    output = model(x, t, context, seq_len, clip_fea=clip_fea, y=y)
                    logging.info(f"Forward pass successful! Output shape: {output[0].shape}")
                    log_memory("After forward pass")
                except Exception as e:
                    logging.error(f"Forward pass failed: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Cleanup
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_memory("After cleanup")
        
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
    
    logging.info("=== Test Complete ===")

if __name__ == "__main__":
    main()
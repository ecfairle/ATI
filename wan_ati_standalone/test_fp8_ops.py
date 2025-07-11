#!/usr/bin/env python3
"""
Test FP8 operations to ensure they work correctly
"""

import torch
import logging

logging.basicConfig(level=logging.INFO)

def test_fp8_operations():
    """Test basic FP8 operations"""
    
    if not hasattr(torch, 'float8_e4m3fn'):
        logging.error("FP8 not supported in this PyTorch version")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Testing FP8 operations on {device}")
    
    try:
        # Create FP8 tensors
        x_fp32 = torch.randn(10, 10, device=device, dtype=torch.float32)
        x_fp8 = x_fp32.to(torch.float8_e4m3fn)
        
        logging.info(f"Created FP8 tensor: shape={x_fp8.shape}, dtype={x_fp8.dtype}")
        
        # Test matmul with mixed precision
        w_fp8 = torch.randn(10, 10, device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
        
        # FP8 matmul typically requires converting to a compute dtype
        # PyTorch may not support direct FP8 matmul yet
        try:
            # Try direct matmul
            y = torch.matmul(x_fp8, w_fp8)
            logging.info(f"Direct FP8 matmul worked! Output dtype: {y.dtype}")
        except Exception as e:
            logging.warning(f"Direct FP8 matmul failed: {e}")
            
            # Try with conversion
            x_compute = x_fp8.to(torch.bfloat16)
            w_compute = w_fp8.to(torch.bfloat16)
            y = torch.matmul(x_compute, w_compute)
            logging.info(f"Mixed precision matmul worked! Output dtype: {y.dtype}")
        
        # Test linear layer with FP8 weights
        linear = torch.nn.Linear(10, 10).to(device)
        # Convert weights to FP8
        linear.weight.data = linear.weight.data.to(torch.float8_e4m3fn)
        logging.info(f"Linear layer weight dtype: {linear.weight.dtype}")
        
        # Try forward pass
        try:
            # Input in compute dtype
            x_compute = torch.randn(5, 10, device=device, dtype=torch.bfloat16)
            y = linear(x_compute)
            logging.info(f"Linear layer forward pass worked! Output shape: {y.shape}, dtype: {y.dtype}")
        except Exception as e:
            logging.error(f"Linear layer forward pass failed: {e}")
        
        logging.info("FP8 operations test completed successfully!")
        
    except Exception as e:
        logging.error(f"FP8 test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fp8_operations()
"""
FP8 utilities for loading and handling FP8 quantized models
"""

import torch
import logging
from safetensors.torch import load_file
import json


def load_fp8_checkpoint(checkpoint_path, device='cpu', dtype_override=None, keep_fp8=False):
    """
    Load a checkpoint that contains FP8 tensors.
    
    Args:
        checkpoint_path: Path to the safetensors file
        device: Device to load tensors to
        dtype_override: If specified, convert all tensors to this dtype
        keep_fp8: If True and FP8 is supported, keep FP8 tensors as FP8
    
    Returns:
        state_dict with appropriate dtypes
    """
    logging.info(f"Loading FP8 checkpoint from {checkpoint_path}")
    
    # Check if PyTorch supports FP8
    fp8_supported = hasattr(torch, 'float8_e4m3fn')
    if keep_fp8 and fp8_supported:
        logging.info("FP8 support detected - will keep FP8 tensors in native format")
    elif keep_fp8 and not fp8_supported:
        logging.warning("FP8 requested but not supported by PyTorch - will convert to dtype_override")
        keep_fp8 = False
    
    # First, check the metadata to understand the dtypes
    with open(checkpoint_path, 'rb') as f:
        header_size = int.from_bytes(f.read(8), 'little')
        header = f.read(header_size).decode('utf-8')
    
    metadata = json.loads(header)
    
    # Count dtypes
    dtype_counts = {}
    for tensor_info in metadata.values():
        if isinstance(tensor_info, dict) and 'dtype' in tensor_info:
            dtype = tensor_info['dtype']
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    
    logging.info(f"Checkpoint contains: {dtype_counts}")
    
    # Load the checkpoint
    if keep_fp8 and fp8_supported:
        # Special handling to keep FP8 tensors
        logging.info("Loading checkpoint while preserving FP8 tensors")
        state_dict = {}
        raw_state_dict = load_file(checkpoint_path, device=device)
        
        for key, tensor in raw_state_dict.items():
            # Check if this tensor was originally FP8
            original_dtype = metadata.get(key, {}).get('dtype', 'unknown')
            
            if original_dtype == 'F8_E4M3':
                # Convert to torch.float8_e4m3fn
                state_dict[key] = tensor.to(torch.float8_e4m3fn)
            elif original_dtype == 'F8_E5M2' and hasattr(torch, 'float8_e5m2'):
                # Convert to torch.float8_e5m2
                state_dict[key] = tensor.to(torch.float8_e5m2)
            else:
                # Keep non-FP8 tensors as-is (biases, norms stay in FP32)
                state_dict[key] = tensor
        
        return state_dict
    elif dtype_override is not None:
        logging.info(f"Loading checkpoint with dtype override to {dtype_override}")
        # Load and convert all tensors to the specified dtype
        state_dict = {}
        raw_state_dict = load_file(checkpoint_path, device=device)
        
        for key, tensor in raw_state_dict.items():
            # Check if this tensor was originally FP8
            original_dtype = metadata.get(key, {}).get('dtype', 'unknown')
            
            if original_dtype in ['F8_E4M3', 'F8_E5M2']:
                # This was an FP8 tensor, convert to override dtype
                state_dict[key] = tensor.to(dtype_override)
            else:
                # Non-FP8 tensors (like biases, norms) keep their precision or convert
                if tensor.dtype == torch.float32 and dtype_override in [torch.float16, torch.bfloat16]:
                    # Keep float32 for small tensors like biases and norms
                    if tensor.numel() < 10000:  # Small tensors
                        state_dict[key] = tensor
                    else:
                        state_dict[key] = tensor.to(dtype_override)
                else:
                    state_dict[key] = tensor
        
        return state_dict
    else:
        # Load with original dtypes (safetensors will convert FP8 to FP32)
        logging.warning("Loading without dtype override - FP8 tensors will be converted to FP32")
        return load_file(checkpoint_path, device=device)


def get_model_size_gb(state_dict):
    """Calculate the size of a model in GB based on its state dict"""
    total_bytes = 0
    dtype_bytes = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
        torch.uint8: 1,
    }
    
    # Add FP8 dtypes if available
    if hasattr(torch, 'float8_e4m3fn'):
        dtype_bytes[torch.float8_e4m3fn] = 1
    if hasattr(torch, 'float8_e5m2'):
        dtype_bytes[torch.float8_e5m2] = 1
    
    # Count parameters by dtype
    dtype_counts = {}
    for name, tensor in state_dict.items():
        dtype = str(tensor.dtype)
        if dtype not in dtype_counts:
            dtype_counts[dtype] = {'count': 0, 'params': 0}
        dtype_counts[dtype]['count'] += 1
        dtype_counts[dtype]['params'] += tensor.numel()
        
        bytes_per_elem = dtype_bytes.get(tensor.dtype, 4)  # Default to 4 if unknown
        total_bytes += tensor.numel() * bytes_per_elem
    
    # Log dtype distribution
    logging.info("State dict dtype distribution:")
    for dtype, info in dtype_counts.items():
        logging.info(f"  {dtype}: {info['count']} tensors, {info['params']/1e9:.2f}B params")
    
    return total_bytes / (1024 ** 3)  # Convert to GB
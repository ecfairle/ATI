"""
Utilities for memory-efficient safetensors loading
"""

import json
import torch
import logging
from typing import Dict, List, Tuple, Optional
import gc
import struct
import mmap


def get_checkpoint_metadata(checkpoint_path: str) -> Dict:
    """
    Read metadata from safetensors file without loading tensors
    """
    with open(checkpoint_path, 'rb') as f:
        # Read header size (first 8 bytes)
        header_size = struct.unpack('<Q', f.read(8))[0]
        # Read header
        header_data = f.read(header_size)
        return json.loads(header_data)


def load_tensor_from_file(checkpoint_path: str, tensor_name: str, metadata: Dict) -> torch.Tensor:
    """
    Load a single tensor from safetensors file
    """
    if tensor_name not in metadata:
        raise KeyError(f"Tensor {tensor_name} not found in checkpoint")
    
    tensor_info = metadata[tensor_name]
    dtype_str = tensor_info['dtype']
    shape = tensor_info['shape']
    start_offset = tensor_info['data_offsets'][0]
    end_offset = tensor_info['data_offsets'][1]
    
    # Map dtype strings to torch dtypes
    dtype_map = {
        'F32': torch.float32,
        'F16': torch.float16,
        'BF16': torch.bfloat16,
        'F8_E4M3': torch.float32,  # Load as float32, will convert later
        'F8_E5M2': torch.float32,  # Load as float32, will convert later
        'I32': torch.int32,
        'I64': torch.int64,
    }
    
    torch_dtype = dtype_map.get(dtype_str, torch.float32)
    
    # Read tensor data
    with open(checkpoint_path, 'rb') as f:
        # Skip to tensor data (header size + 8 bytes for header size itself)
        f.seek(8)
        header_size = struct.unpack('<Q', f.read(8))[0]
        f.seek(8 + header_size + start_offset)
        
        # Read tensor bytes
        num_bytes = end_offset - start_offset
        tensor_bytes = f.read(num_bytes)
    
    # Create tensor from bytes
    if dtype_str in ['F8_E4M3', 'F8_E5M2']:
        # For FP8, safetensors stores as uint8, we need to load and convert
        # First load the raw bytes
        import numpy as np
        numpy_array = np.frombuffer(tensor_bytes, dtype=np.uint8).reshape(shape)
        # Convert to float32 tensor first (safetensors' default behavior)
        tensor = torch.from_numpy(numpy_array).to(torch.float32)
        # Scale the values appropriately (FP8 range)
        # This is a simplified conversion - safetensors has the actual conversion logic
        # For now, we'll use safetensors' load_file for FP8 tensors
        return None  # Signal to use fallback
    else:
        tensor = torch.frombuffer(tensor_bytes, dtype=torch_dtype)
        tensor = tensor.reshape(shape)
    
    return tensor.clone()  # Clone to ensure it's not tied to the buffer


def load_checkpoint_in_chunks(
    checkpoint_path: str, 
    device: str = 'cpu',
    keep_fp8: bool = True,
    chunk_size: int = 50
) -> Dict[str, torch.Tensor]:
    """
    Load checkpoint in chunks to minimize memory usage
    """
    logging.info(f"Loading checkpoint in chunks from {checkpoint_path}")
    
    # Get metadata
    metadata = get_checkpoint_metadata(checkpoint_path)
    
    # Separate tensors by size
    large_tensors = []
    small_tensors = []
    
    for name, info in metadata.items():
        if not isinstance(info, dict) or 'shape' not in info:
            continue
            
        # Calculate tensor size
        shape = info['shape']
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        
        if num_elements > 10_000_000:  # 10M parameters
            large_tensors.append((name, info))
        else:
            small_tensors.append((name, info))
    
    logging.info(f"Found {len(large_tensors)} large tensors and {len(small_tensors)} small tensors")
    
    state_dict = {}
    
    # Process large tensors one at a time
    for tensor_name, tensor_info in large_tensors:
        logging.info(f"Loading large tensor: {tensor_name} (shape: {tensor_info['shape']})")
        
        # Load tensor
        tensor = load_tensor_from_file(checkpoint_path, tensor_name, metadata)
        
        # Convert dtype if needed
        if keep_fp8 and tensor_info['dtype'] == 'F8_E4M3' and hasattr(torch, 'float8_e4m3fn'):
            tensor = tensor.to(torch.float8_e4m3fn)
            logging.info(f"  Converted to FP8: {tensor.dtype}")
        elif tensor_info['dtype'] in ['F8_E4M3', 'F8_E5M2']:
            # If we can't keep FP8, convert to bfloat16
            tensor = tensor.to(torch.bfloat16)
            logging.info(f"  Converted to BF16: {tensor.dtype}")
        
        # Move to device if needed
        if device != 'cpu':
            tensor = tensor.to(device)
        
        state_dict[tensor_name] = tensor
        del tensor
        gc.collect()
        
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Process small tensors in batches
    logging.info(f"Loading {len(small_tensors)} small tensors")
    for i in range(0, len(small_tensors), chunk_size):
        batch = small_tensors[i:i + chunk_size]
        
        for tensor_name, tensor_info in batch:
            tensor = load_tensor_from_file(checkpoint_path, tensor_name, metadata)
            
            # Small tensors (biases, norms) typically stay in float32
            if device != 'cpu':
                tensor = tensor.to(device)
            
            state_dict[tensor_name] = tensor
        
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    logging.info(f"Loaded {len(state_dict)} tensors successfully")
    return state_dict
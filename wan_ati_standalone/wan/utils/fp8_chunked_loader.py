"""
Simplified chunked loader using safetensors but with better memory management
"""

import torch
import logging
import gc
from safetensors import safe_open
from typing import Dict, List, Optional


def load_fp8_checkpoint_chunked(
    checkpoint_path: str,
    device: str = 'cpu', 
    keep_fp8: bool = True,
    max_memory_gb: float = 10.0,
    metadata: Optional[Dict] = None,
    dtype_override: Optional[torch.dtype] = None
) -> Dict[str, torch.Tensor]:
    """
    Load checkpoint with memory-efficient processing
    
    Args:
        checkpoint_path: Path to safetensors file
        device: Target device ('cpu' or 'cuda')
        keep_fp8: Whether to keep tensors in FP8 format
        max_memory_gb: Maximum memory to use at once (in GB)
        metadata: Optional metadata dict with tensor dtype info
        dtype_override: Optional dtype to convert tensors to
    """
    logging.info(f"Loading checkpoint with chunked processing (max memory: {max_memory_gb}GB)")
    
    state_dict = {}
    current_memory_bytes = 0
    max_memory_bytes = int(max_memory_gb * 1024**3)
    
    # Track which tensors to process
    pending_tensors = []
    
    # First pass: get tensor info without loading data
    with safe_open(checkpoint_path, framework="pt", device='cpu') as f:
        # Get metadata to check tensor info
        tensors = f.keys()
        
        for key in tensors:
            # Get tensor metadata without loading the actual tensor
            # We'll estimate size based on typical dtypes
            # Most weights will be FP8 (1 byte), biases FP32 (4 bytes)
            
            # Try to infer from name
            if 'weight' in key and 'norm' not in key:
                # Likely a weight matrix, will be FP8
                bytes_per_element = 1 if keep_fp8 else 4
            else:
                # Likely bias or norm, stays FP32
                bytes_per_element = 4
            
            # Estimate tensor size (we'll refine this during actual loading)
            # For now, assume average tensor is 100M parameters
            estimated_bytes = 100_000_000 * bytes_per_element
            pending_tensors.append((key, estimated_bytes))
    
    # Sort by size (process large tensors first)
    pending_tensors.sort(key=lambda x: x[1], reverse=True)
    
    # Process tensors in batches
    batch = []
    batch_bytes = 0
    
    for tensor_name, tensor_bytes in pending_tensors:
        # Check if adding this tensor would exceed memory limit
        if batch and (batch_bytes + tensor_bytes > max_memory_bytes):
            # Process current batch
            _process_tensor_batch(checkpoint_path, batch, state_dict, device, keep_fp8, metadata, dtype_override)
            batch = []
            batch_bytes = 0
            
        batch.append(tensor_name)
        batch_bytes += tensor_bytes
    
    # Process final batch
    if batch:
        _process_tensor_batch(checkpoint_path, batch, state_dict, device, keep_fp8, metadata, dtype_override)
    
    logging.info(f"Loaded {len(state_dict)} tensors successfully")
    return state_dict


def _process_tensor_batch(
    checkpoint_path: str,
    tensor_names: List[str],
    state_dict: Dict[str, torch.Tensor],
    device: str,
    keep_fp8: bool,
    metadata: Optional[Dict] = None,
    dtype_override: Optional[torch.dtype] = None
):
    """Process a batch of tensors"""
    logging.info(f"Processing batch of {len(tensor_names)} tensors")
    
    # Load tensors in this batch
    with safe_open(checkpoint_path, framework="pt", device='cpu') as f:
        for tensor_name in tensor_names:
            # Load tensor
            tensor = f.get_tensor(tensor_name)
            
            # Check original dtype from metadata if available
            original_dtype = 'F32'  # Default
            if metadata and tensor_name in metadata:
                original_dtype = metadata[tensor_name].get('dtype', 'F32')
            
            # Handle dtype conversion
            if dtype_override is not None:
                # Apply dtype override
                if original_dtype in ['F8_E4M3', 'F8_E5M2']:
                    # This was an FP8 tensor, convert to override dtype
                    tensor = tensor.to(dtype_override)
                    logging.debug(f"  {tensor_name}: FP8 -> {dtype_override}")
                else:
                    # Non-FP8 tensors (like biases, norms) - handle based on size
                    if tensor.dtype == torch.float32 and dtype_override in [torch.float16, torch.bfloat16]:
                        # Keep float32 for small tensors like biases and norms
                        if tensor.numel() < 10000:  # Small tensors
                            logging.debug(f"  {tensor_name}: keeping FP32 (small tensor)")
                        else:
                            tensor = tensor.to(dtype_override)
                            logging.debug(f"  {tensor_name}: FP32 -> {dtype_override}")
            elif keep_fp8 and original_dtype == 'F8_E4M3' and hasattr(torch, 'float8_e4m3fn'):
                # Safetensors loads FP8 as float32, convert back to FP8
                tensor = tensor.to(torch.float8_e4m3fn)
                logging.debug(f"  {tensor_name}: converted to FP8")
            elif original_dtype in ['F8_E4M3', 'F8_E5M2'] and not keep_fp8:
                # Convert to bfloat16 if not keeping FP8
                tensor = tensor.to(torch.bfloat16)
                logging.debug(f"  {tensor_name}: converted to BF16")
            
            # Move to device
            if device != 'cpu':
                tensor = tensor.to(device)
            
            state_dict[tensor_name] = tensor
    
    # Force garbage collection
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
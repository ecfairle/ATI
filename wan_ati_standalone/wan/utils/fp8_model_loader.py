"""
Custom model loader that preserves FP8 dtypes
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any


def load_state_dict_fp8(model: nn.Module, state_dict: Dict[str, torch.Tensor], strict: bool = True):
    """
    Load state dict while preserving FP8 dtypes.
    
    This function bypasses PyTorch's automatic dtype conversion by directly
    assigning parameter data.
    """
    # First check if model has meta tensors
    has_meta = any(p.is_meta for p in model.parameters())
    if has_meta:
        # If model has meta tensors, we need to materialize it first
        # This should have been done by the caller, but let's be safe
        device = next(iter(state_dict.values())).device
        model = model.to(device)
    
    model_state = model.state_dict()
    
    # Track what we're loading
    loaded_keys = set()
    skipped_keys = set()
    dtype_conversions = {}
    
    for name, param in state_dict.items():
        if name in model_state:
            model_param = model_state[name]
            
            # Check if shapes match
            if model_param.shape != param.shape:
                raise RuntimeError(f"Shape mismatch for {name}: model has {model_param.shape}, checkpoint has {param.shape}")
            
            # Get the actual parameter from the model
            module = model
            parts = name.split('.')
            for part in parts[:-1]:
                module = getattr(module, part)
            param_name = parts[-1]
            
            if hasattr(module, param_name):
                old_param = getattr(module, param_name)
                
                # Preserve FP8 dtype if present
                if hasattr(torch, 'float8_e4m3fn') and param.dtype == torch.float8_e4m3fn:
                    # For FP8 parameters, create a new parameter with FP8 data
                    if isinstance(old_param, nn.Parameter):
                        new_param = nn.Parameter(param.data.clone(), requires_grad=old_param.requires_grad)
                        setattr(module, param_name, new_param)
                        dtype_conversions[name] = f"{old_param.dtype} -> {param.dtype}"
                    else:
                        # For buffers
                        setattr(module, param_name, param.data.clone())
                        dtype_conversions[name] = f"buffer: {old_param.dtype} -> {param.dtype}"
                else:
                    # For non-FP8 parameters, use normal assignment
                    if isinstance(old_param, nn.Parameter):
                        old_param.data.copy_(param.data)
                    else:
                        setattr(module, param_name, param.data.clone())
                
                loaded_keys.add(name)
        else:
            skipped_keys.add(name)
    
    # Check for missing keys if strict
    missing_keys = set(model_state.keys()) - loaded_keys
    if strict and (missing_keys or skipped_keys):
        raise RuntimeError(f"Error in loading state dict. Missing keys: {missing_keys}, Unexpected keys: {skipped_keys}")
    
    # Log statistics
    logging.info(f"Loaded {len(loaded_keys)} parameters")
    if dtype_conversions:
        logging.info(f"Preserved FP8 for {len(dtype_conversions)} parameters")
        # Show a few examples
        for i, (name, conversion) in enumerate(dtype_conversions.items()):
            if i < 5:
                logging.info(f"  {name}: {conversion}")
            else:
                break
        if len(dtype_conversions) > 5:
            logging.info(f"  ... and {len(dtype_conversions) - 5} more")
    
    return model


def move_model_to_device_fp8(model: nn.Module, device: str):
    """
    Move model to device while preserving FP8 dtypes.
    """
    if not torch.cuda.is_available() and 'cuda' in device:
        logging.warning("CUDA not available, keeping model on CPU")
        return model
    
    logging.info(f"Moving model to {device} while preserving FP8 dtypes...")
    
    fp8_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        if hasattr(torch, 'float8_e4m3fn') and param.dtype == torch.float8_e4m3fn:
            # Move FP8 parameters to device while preserving dtype
            param.data = param.data.to(device=device)
            fp8_params += 1
        else:
            # Move other parameters normally
            param.data = param.data.to(device=device)
    
    # Move buffers
    for name, buffer in model.named_buffers():
        if buffer is not None:
            # Get the buffer's parent module and attribute name
            module = model
            parts = name.split('.')
            for part in parts[:-1]:
                module = getattr(module, part)
            buffer_name = parts[-1]
            
            # Move buffer to device
            setattr(module, buffer_name, buffer.to(device=device))
    
    if fp8_params > 0:
        logging.info(f"Moved {fp8_params}/{total_params} FP8 parameters to {device}")
    
    return model
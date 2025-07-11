"""
Mixin class to add FP8 support to PyTorch models
"""

import torch
import torch.nn as nn
import logging


class FP8ModelMixin:
    """
    Mixin to handle FP8 tensors properly in PyTorch models.
    Prevents automatic dtype conversion when moving models to GPU.
    """
    
    def _apply(self, fn):
        """
        Override _apply to preserve FP8 dtypes when moving to GPU
        """
        def _apply_fn(t):
            # Check if tensor is FP8
            if hasattr(torch, 'float8_e4m3fn') and t.dtype == torch.float8_e4m3fn:
                # For FP8 tensors, only change device, not dtype
                if 'cuda' in str(fn):
                    # Extract device from the lambda function
                    # This is a bit hacky but works for the common case
                    try:
                        # Try to move to cuda while preserving dtype
                        device = 'cuda' if t.is_cpu else t.device
                        return t.to(device=device)
                    except:
                        # Fallback to original function
                        return fn(t)
                else:
                    return fn(t)
            else:
                # For non-FP8 tensors, use original function
                return fn(t)
        
        # Apply to all parameters and buffers
        for module in self.children():
            module._apply(fn)
        
        for key, param in self._parameters.items():
            if param is not None:
                with torch.no_grad():
                    self._parameters[key] = nn.Parameter(_apply_fn(param.data), param.requires_grad)
        
        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = _apply_fn(buf)
        
        return self
    
    def cuda(self, device=None):
        """
        Override cuda() to properly handle FP8 tensors
        """
        def cuda_fn(t):
            if hasattr(torch, 'float8_e4m3fn') and t.dtype == torch.float8_e4m3fn:
                # Keep FP8 dtype when moving to CUDA
                return t.cuda(device) if device is None else t.to(device=device)
            else:
                # Use default behavior for other dtypes
                return t.cuda(device)
        
        return self._apply(cuda_fn)
    
    def to(self, *args, **kwargs):
        """
        Override to() to handle FP8 tensors properly
        """
        # Parse arguments
        device = None
        dtype = None
        
        if args:
            arg = args[0]
            if isinstance(arg, torch.dtype):
                dtype = arg
            elif isinstance(arg, (str, torch.device)):
                device = arg
            elif isinstance(arg, torch.Tensor):
                device = arg.device
                dtype = arg.dtype
        
        device = kwargs.get('device', device)
        dtype = kwargs.get('dtype', dtype)
        
        def to_fn(t):
            if hasattr(torch, 'float8_e4m3fn') and t.dtype == torch.float8_e4m3fn:
                # For FP8 tensors, only change device if specified, keep dtype
                if device is not None:
                    return t.to(device=device)
                else:
                    return t
            else:
                # Use normal to() for non-FP8 tensors
                if device is not None and dtype is not None:
                    return t.to(device=device, dtype=dtype)
                elif device is not None:
                    return t.to(device=device)
                elif dtype is not None:
                    return t.to(dtype=dtype)
                else:
                    return t
        
        return self._apply(to_fn)
    
    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict to preserve FP8 dtypes
        """
        # First, ensure model parameters can accept FP8 dtypes
        for name, param in state_dict.items():
            if hasattr(torch, 'float8_e4m3fn') and param.dtype == torch.float8_e4m3fn:
                # Find the corresponding parameter in the model
                parts = name.split('.')
                module = self
                for part in parts[:-1]:
                    module = getattr(module, part)
                
                param_name = parts[-1]
                if hasattr(module, param_name):
                    # Replace the parameter with one that accepts FP8
                    old_param = getattr(module, param_name)
                    if isinstance(old_param, nn.Parameter):
                        # Create new parameter with FP8 data
                        new_param = nn.Parameter(param.data, requires_grad=old_param.requires_grad)
                        setattr(module, param_name, new_param)
        
        # Now load the state dict normally
        return super().load_state_dict(state_dict, strict=strict)


def make_fp8_compatible(model_class):
    """
    Create a new class that inherits from both the model class and FP8ModelMixin
    """
    class FP8CompatibleModel(FP8ModelMixin, model_class):
        pass
    
    FP8CompatibleModel.__name__ = f"FP8{model_class.__name__}"
    return FP8CompatibleModel
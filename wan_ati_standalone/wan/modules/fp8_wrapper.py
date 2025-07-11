"""
Simple FP8 model wrapper that handles dtype conversion at compute time
"""

import torch
import torch.nn as nn
import logging


class FP8ModelWrapper(nn.Module):
    """
    Wrapper that keeps model weights in FP8 but handles conversion for computation
    """
    
    def __init__(self, model, compute_dtype=torch.bfloat16):
        super().__init__()
        self.model = model
        self.compute_dtype = compute_dtype
        self._fp8_converted = False
        
    def _convert_to_fp8(self):
        """Convert all large weight tensors to FP8"""
        if self._fp8_converted:
            return
            
        if not hasattr(torch, 'float8_e4m3fn'):
            logging.warning("FP8 not supported, keeping original dtypes")
            return
            
        logging.info("Converting model weights to FP8...")
        total_params = 0
        fp8_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            
            # Only convert large tensors (weights, not biases/norms)
            if param.numel() > 10000 and param.dim() >= 2:
                # Convert to FP8
                param.data = param.data.to(torch.float8_e4m3fn)
                fp8_params += param.numel()
        
        logging.info(f"Converted {fp8_params/total_params*100:.1f}% of parameters to FP8")
        self._fp8_converted = True
    
    def forward(self, *args, **kwargs):
        # Ensure model is in FP8
        self._convert_to_fp8()
        
        # Create a compute context that temporarily converts weights
        with FP8ComputeContext(self.model, self.compute_dtype):
            return self.model(*args, **kwargs)
    
    def to(self, *args, **kwargs):
        # Move the wrapped model
        self.model = self.model.to(*args, **kwargs)
        return self
    
    def cuda(self, device=None):
        self.model = self.model.cuda(device)
        return self
    
    def eval(self):
        self.model.eval()
        return self
    
    def requires_grad_(self, requires_grad):
        self.model.requires_grad_(requires_grad)
        return self


class FP8ComputeContext:
    """Context manager that temporarily converts FP8 weights to compute dtype"""
    
    def __init__(self, model, compute_dtype=torch.bfloat16):
        self.model = model
        self.compute_dtype = compute_dtype
        self.fp8_params = []
        
    def __enter__(self):
        # Find and temporarily convert FP8 parameters
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                if hasattr(torch, 'float8_e4m3fn') and module.weight.dtype == torch.float8_e4m3fn:
                    # Store original weight and convert to compute dtype
                    self.fp8_params.append((module, 'weight', module.weight))
                    module.weight = nn.Parameter(
                        module.weight.to(self.compute_dtype),
                        requires_grad=module.weight.requires_grad
                    )
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore FP8 weights
        for module, attr_name, original_param in self.fp8_params:
            setattr(module, attr_name, original_param)
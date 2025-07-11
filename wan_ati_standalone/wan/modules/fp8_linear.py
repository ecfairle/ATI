"""
FP8-aware Linear layer that handles mixed precision computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FP8Linear(nn.Module):
    """
    Linear layer that supports FP8 weights with automatic mixed precision computation
    """
    
    def __init__(self, in_features, out_features, bias=True, compute_dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        
        # Initialize parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights (will be overwritten when loading checkpoint)
        self.reset_parameters()
    
    def reset_parameters(self):
        # Standard initialization
        nn.init.kaiming_uniform_(self.weight, a=torch.nn.init.calculate_gain('linear'))
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, input):
        # Check if weight is in FP8
        if hasattr(torch, 'float8_e4m3fn') and self.weight.dtype == torch.float8_e4m3fn:
            # Convert to compute dtype for the operation
            weight = self.weight.to(self.compute_dtype)
        else:
            weight = self.weight
        
        # Ensure input is in compute dtype
        if input.dtype != self.compute_dtype:
            input = input.to(self.compute_dtype)
        
        return F.linear(input, weight, self.bias)
    
    @classmethod
    def from_linear(cls, linear_layer, compute_dtype=torch.bfloat16):
        """Create FP8Linear from existing nn.Linear layer"""
        fp8_linear = cls(
            linear_layer.in_features,
            linear_layer.out_features,
            linear_layer.bias is not None,
            compute_dtype
        )
        fp8_linear.weight = linear_layer.weight
        if linear_layer.bias is not None:
            fp8_linear.bias = linear_layer.bias
        return fp8_linear


def replace_linear_with_fp8(module, compute_dtype=torch.bfloat16):
    """
    Recursively replace nn.Linear layers with FP8Linear layers in a module
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Check if the linear layer has FP8 weights
            if hasattr(torch, 'float8_e4m3fn') and child.weight.dtype == torch.float8_e4m3fn:
                fp8_linear = FP8Linear.from_linear(child, compute_dtype)
                setattr(module, name, fp8_linear)
        else:
            # Recursively apply to child modules
            replace_linear_with_fp8(child, compute_dtype)
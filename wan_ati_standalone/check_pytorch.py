#!/usr/bin/env python3
import torch
import sys

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if hasattr(torch.version, 'cuda'):
    print("PyTorch CUDA version:", torch.version.cuda)
else:
    print("PyTorch CUDA version: Not found (CPU-only build?)")

print("\nCUDA arch list:", torch.cuda.get_arch_list() if hasattr(torch.cuda, 'get_arch_list') else "Not available")

# Try to get more detailed error
try:
    torch.cuda.init()
    print("\nCUDA init successful")
except Exception as e:
    print(f"\nCUDA init error: {e}")

# Check if this is a CUDA version issue
print("\nChecking CUDA runtime...")
try:
    torch.cuda._check_driver()
    print("Driver check passed")
except Exception as e:
    print(f"Driver check failed: {e}")
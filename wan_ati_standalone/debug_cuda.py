#!/usr/bin/env python3
"""Debug script to check CUDA availability and configuration."""

import torch
import os
import subprocess

print("=== CUDA Debugging Information ===\n")

# Check PyTorch CUDA availability
print("1. PyTorch CUDA Status:")
print(f"   - torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"   - torch.cuda.device_count(): {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"   - Current device: {torch.cuda.current_device()}")
    print(f"   - Device name: {torch.cuda.get_device_name(0)}")
    print(f"   - CUDA version: {torch.version.cuda}")
print(f"   - PyTorch version: {torch.__version__}")

# Check environment variables
print("\n2. CUDA Environment Variables:")
cuda_vars = ['CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'CUDA_PATH', 'LD_LIBRARY_PATH']
for var in cuda_vars:
    value = os.environ.get(var, 'Not set')
    print(f"   - {var}: {value}")

# Try to get nvidia-smi output
print("\n3. NVIDIA Driver Information:")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("   nvidia-smi output:")
        print("   " + "\n   ".join(result.stdout.split('\n')[:10]))  # First 10 lines
    else:
        print("   nvidia-smi command failed")
except FileNotFoundError:
    print("   nvidia-smi not found in PATH")

# Check for common issues
print("\n4. Common Issues Check:")

# Check if running in container/WSL
if os.path.exists('/.dockerenv'):
    print("   - Running in Docker container")
if 'WSL' in os.environ.get('WSL_DISTRO_NAME', ''):
    print("   - Running in WSL")

# Check PyTorch CUDA build
if not torch.cuda.is_available() and hasattr(torch.version, 'cuda'):
    if torch.version.cuda:
        print(f"   - PyTorch built with CUDA {torch.version.cuda} but CUDA not available")
        print("   - This might indicate a driver/library mismatch")
    else:
        print("   - PyTorch built WITHOUT CUDA support (CPU-only)")

print("\n5. Troubleshooting suggestions:")
if not torch.cuda.is_available():
    print("   - Verify NVIDIA drivers are installed: sudo apt install nvidia-driver-XXX")
    print("   - Check if GPU is visible: lspci | grep -i nvidia")
    print("   - Ensure PyTorch is installed with CUDA support")
    print("   - Try setting CUDA_VISIBLE_DEVICES=0")
    print("   - If in container, ensure --gpus all flag is used")
    print("   - If in WSL2, ensure GPU passthrough is configured")
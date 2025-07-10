#!/usr/bin/env python3
"""Script to diagnose and suggest fixes for CUDA issues."""

import os
import subprocess

print("=== CUDA Version Compatibility Check ===\n")

# Get CUDA driver version from nvidia-smi
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                          capture_output=True, text=True)
    driver_version = result.stdout.strip()
    print(f"NVIDIA Driver Version: {driver_version}")
except:
    print("Could not get driver version")

# Get CUDA runtime version
print(f"System CUDA Version: 12.9 (from nvidia-smi)")
print(f"PyTorch CUDA Version: 12.8")

print("\n=== Diagnosis ===")
print("The issue is likely due to PyTorch being compiled for CUDA 12.8 while your system has CUDA 12.9")
print("CUDA 12.9 is very new and PyTorch might not have full support yet.")

print("\n=== Solutions ===")
print("1. Install PyTorch with CUDA 12.1 (most stable):")
print("   pip uninstall torch torchvision torchaudio")
print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

print("\n2. Or try PyTorch nightly with CUDA 12.4:")
print("   pip uninstall torch torchvision torchaudio")
print("   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124")

print("\n3. Set environment variables (temporary fix):")
print("   export CUDA_HOME=/usr/local/cuda")
print("   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")
print("   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512")

print("\n4. Check LD_LIBRARY_PATH:")
ld_path = os.environ.get('LD_LIBRARY_PATH', 'Not set')
print(f"   Current LD_LIBRARY_PATH: {ld_path}")

if '/usr/local/cuda/lib64' not in ld_path:
    print("   WARNING: CUDA libraries not in LD_LIBRARY_PATH!")
    
print("\n5. Try running with specific CUDA device:")
print("   export CUDA_VISIBLE_DEVICES=0")
print("   export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync")
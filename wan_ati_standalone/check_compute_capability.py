#!/usr/bin/env python3
"""Check GPU compute capability and PyTorch support."""

import torch
import subprocess

print("=== GPU Compute Capability Check ===\n")

# Get GPU info from nvidia-smi
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv'], 
                          capture_output=True, text=True)
    print("GPU Information from nvidia-smi:")
    print(result.stdout)
except:
    print("Could not query GPU info")

print("\nPyTorch Information:")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"\nCUDA device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nDevice {i}: {props.name}")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {props.total_memory / 1024**3:.1f} GB")
else:
    print("\nCUDA is not available to check device properties")

print("\n=== RTX 5090 Requirements ===")
print("RTX 5090 (Ada Lovelace) requires:")
print("- Compute Capability: 8.9")
print("- CUDA 12.0 or higher")
print("- PyTorch compiled with sm_89 support")

print("\n=== Recommended Solutions ===")
print("1. Install PyTorch nightly (latest CUDA support):")
print("   pip uninstall torch torchvision torchaudio")
print("   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124")

print("\n2. Or install PyTorch 2.5.1 with CUDA 12.4:")
print("   pip uninstall torch torchvision torchaudio")  
print("   pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124")

print("\n3. If issues persist, try the test channel:")
print("   pip uninstall torch torchvision torchaudio")
print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu124")

print("\n4. Check if PyTorch has sm_89 support:")
if torch.cuda.is_available():
    try:
        # This will show supported architectures
        import torch._C
        print(f"   Supported CUDA architectures: {torch.cuda.get_arch_list()}")
    except:
        print("   Could not check supported architectures")
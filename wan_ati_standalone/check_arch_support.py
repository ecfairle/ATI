#!/usr/bin/env python3
"""Check PyTorch CUDA architecture support."""

import torch
import subprocess

print("=== PyTorch CUDA Architecture Support ===\n")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")

# Check supported architectures
if torch.cuda.is_available():
    arch_list = torch.cuda.get_arch_list()
    print(f"\nSupported CUDA architectures: {arch_list}")
    
    # Parse architectures
    print("\nParsed architectures:")
    for arch in arch_list:
        if arch.startswith('sm_'):
            sm_version = arch[3:]
            major = int(sm_version[0])
            minor = int(sm_version[1])
            print(f"  - sm_{sm_version} (Compute Capability {major}.{minor})")
    
    # Check if sm_89 is supported
    if 'sm_89' in arch_list:
        print("\n✓ RTX 5090 (sm_89) IS supported!")
    else:
        print("\n✗ RTX 5090 (sm_89) is NOT supported!")
        print("\nYour options:")
        print("1. Docker with latest PyTorch:")
        print("   docker run --gpus all -it pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
        print("\n2. Build PyTorch from source:")
        print("   TORCH_CUDA_ARCH_LIST='8.9' pip install ninja")
        print("   git clone --recursive https://github.com/pytorch/pytorch")
        print("   cd pytorch && python setup.py install")
        print("\n3. Use CPU for testing (slow):")
        print("   Modify run_wan_ati.py to use device='cpu'")
        print("\n4. Try WSL2 with Windows PyTorch (if dual-booting)")

# Alternative: Check for specific nightly builds
print("\n\n=== Alternative: Specific Nightly Builds ===")
print("Try these specific builds that might have sm_89 support:")
print("\n1. Latest nightly (force reinstall everything):")
print("   pip uninstall torch torchvision torchaudio -y")
print("   pip install ninja")  # For better compilation
print("   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124 --force-reinstall --upgrade")
print("   pip install --pre torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124 --no-deps")
print("   pip install numpy pillow")  # Install deps separately

print("\n2. Or use conda with cuda-toolkit:")
print("   conda create -n wan python=3.11")
print("   conda activate wan")
print("   conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch-nightly -c nvidia")

# Get actual GPU compute capability
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], 
                          capture_output=True, text=True)
    compute_cap = result.stdout.strip()
    print(f"\n\nYour GPU compute capability: {compute_cap}")
    if compute_cap == "8.9":
        print("Confirmed: You have an RTX 5090 or similar Ada Lovelace GPU")
except:
    pass
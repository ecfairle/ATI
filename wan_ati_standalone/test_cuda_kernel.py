#!/usr/bin/env python3
"""Test basic CUDA operations to identify kernel support issues."""

import torch
import os

# Set debugging environment variable
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print("=== CUDA Kernel Test ===\n")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    # Get device properties
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    print(f"\nGPU: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")
    print(f"CUDA Capability: sm_{props.major}{props.minor}")
    
    # Check supported architectures
    print(f"\nPyTorch built for architectures: {torch.cuda.get_arch_list()}")
    
    # Test 1: Simple tensor creation
    print("\n1. Testing tensor creation on GPU...")
    try:
        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
        print(f"   Success: {x}")
    except Exception as e:
        print(f"   Failed: {e}")
    
    # Test 2: Simple arithmetic
    print("\n2. Testing basic arithmetic...")
    try:
        a = torch.randn(3, 3).cuda()
        b = torch.randn(3, 3).cuda()
        c = a + b
        print(f"   Success: Shape {c.shape}")
    except Exception as e:
        print(f"   Failed: {e}")
    
    # Test 3: Matrix multiplication (often fails with kernel issues)
    print("\n3. Testing matrix multiplication...")
    try:
        a = torch.randn(10, 10).cuda()
        b = torch.randn(10, 10).cuda()
        c = torch.matmul(a, b)
        print(f"   Success: Shape {c.shape}")
    except Exception as e:
        print(f"   Failed: {e}")
    
    # Test 4: Conv2d (common failure point)
    print("\n4. Testing Conv2d operation...")
    try:
        conv = torch.nn.Conv2d(3, 16, 3).cuda()
        x = torch.randn(1, 3, 32, 32).cuda()
        y = conv(x)
        print(f"   Success: Shape {y.shape}")
    except Exception as e:
        print(f"   Failed: {e}")
    
    # Test 5: Check if flash attention would work
    print("\n5. Testing attention-like operation...")
    try:
        q = torch.randn(1, 8, 64, 64).cuda()
        k = torch.randn(1, 8, 64, 64).cuda()
        v = torch.randn(1, 8, 64, 64).cuda()
        scores = torch.matmul(q, k.transpose(-2, -1))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        print(f"   Success: Shape {out.shape}")
    except Exception as e:
        print(f"   Failed: {e}")

print("\n=== Recommendations ===")
if 'sm_89' not in torch.cuda.get_arch_list():
    print("ERROR: PyTorch was NOT compiled with sm_89 support (RTX 5090)")
    print("\nYou need PyTorch compiled specifically for RTX 5090.")
    print("Options:")
    print("1. Install PyTorch nightly:")
    print("   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124 --force-reinstall")
    print("\n2. Build PyTorch from source with TORCH_CUDA_ARCH_LIST='8.9'")
    print("\n3. Use Docker image with latest PyTorch:")
    print("   docker pull pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
else:
    print("PyTorch has sm_89 support. The issue might be elsewhere.")
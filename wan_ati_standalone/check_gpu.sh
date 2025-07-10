#!/bin/bash
echo "=== GPU and CUDA Diagnostics ==="
echo

echo "1. Check for NVIDIA GPU:"
lspci | grep -i nvidia
echo

echo "2. Check NVIDIA driver:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not found"
fi
echo

echo "3. Check kernel modules:"
lsmod | grep nvidia
echo

echo "4. Check CUDA libraries:"
ldconfig -p | grep cuda
echo

echo "5. Check /dev/nvidia* devices:"
ls -la /dev/nvidia* 2>/dev/null || echo "No /dev/nvidia* devices found"
echo

echo "6. Check dmesg for NVIDIA errors:"
dmesg | grep -i nvidia | tail -10
echo

echo "7. WSL specific check:"
if [ -f /proc/version ]; then
    if grep -qi microsoft /proc/version; then
        echo "Running in WSL"
        echo "WSL version:"
        wsl.exe --version 2>/dev/null || echo "Could not determine WSL version"
        
        echo
        echo "Check Windows nvidia-smi:"
        nvidia-smi.exe 2>/dev/null || echo "Windows nvidia-smi not accessible"
    else
        echo "Not running in WSL"
    fi
fi
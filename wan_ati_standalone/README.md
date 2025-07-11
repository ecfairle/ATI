# WAN ATI Standalone - Image-to-Video Generation

This is a minimal, self-contained implementation of the WAN ATI (Animate Through Imagination) model for image-to-video generation with motion trajectory control.

## Overview

WAN ATI is a 14B parameter diffusion transformer model that generates high-quality videos from single images, guided by text prompts and motion trajectories. This standalone version removes distributed training and other complex features to focus on single-GPU inference.

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Download the model checkpoints from the official repository and place them in a directory. You'll need:
   - `Wan2_1-I2V-ATI-14B_fp8_e4m3fn.safetensors` - Main model weights
   - `models_t5_umt5-xxl-enc-bf16.pth` - T5 text encoder
   - `google/umt5-xxl` - T5 tokenizer (will be downloaded automatically)
   - `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` - CLIP visual encoder
   - `xlm-roberta-large` - CLIP tokenizer (will be downloaded automatically)
   - `Wan2.1_VAE.pth` - VAE model

## Usage

### Basic Example

Run the sample bear video generation:

```bash
python run_wan_ati.py --checkpoint_dir /path/to/checkpoints
```

This will:
- Load the sample bear image from `examples/images/bear.jpg`
- Use the corresponding motion trajectory from `examples/tracks/bear.pth`
- Generate an 81-frame video at 720p resolution
- Save the output as `wan_ati_output_TIMESTAMP.mp4`

### Custom Inputs

```bash
python run_wan_ati.py \
    --checkpoint_dir /path/to/checkpoints \
    --image /path/to/your/image.jpg \
    --tracks /path/to/your/tracks.pth \
    --prompt "Your video description here" \
    --output my_video.mp4 \
    --sampling_steps 40 \
    --guide_scale 5.0 \
    --seed 42
```

### Parameters

- `--checkpoint_dir`: Path to directory containing model checkpoints (required)
- `--image`: Input image path (default: sample bear image)
- `--tracks`: Motion trajectory file path (default: sample bear tracks)
- `--prompt`: Text description for the video (default: bear description)
- `--output`: Output video path (default: auto-generated)
- `--sampling_steps`: Number of diffusion steps (default: 40, higher = better quality but slower)
- `--guide_scale`: Classifier-free guidance scale (default: 5.0, higher = stronger prompt adherence)
- `--seed`: Random seed for reproducibility (default: 42)
- `--device`: CUDA device to use (default: cuda:0)

## Expected Output

The model generates:
- 81 frames (approximately 3.4 seconds at 24 FPS)
- 720p resolution (or adaptive based on input image aspect ratio)
- MP4 format video with H.264 encoding

## Motion Trajectories

Motion trajectory files (`.pth` format) contain:
- Shape: [121, H, W, 3] - 121 time steps, height, width, and (x, y, visibility)
- Coordinates normalized to image dimensions
- Automatically downsampled to 81 frames for model input

## System Requirements

- GPU: NVIDIA GPU with at least 24GB VRAM (e.g., RTX 3090, A5000)
- RAM: 32GB+ recommended
- Disk: ~50GB for model checkpoints

## Notes

- The model uses FP16/BF16 precision by default for efficiency
- Model components are offloaded to CPU between steps to reduce VRAM usage
- First run may be slower due to model compilation
- Generation takes approximately 2-5 minutes depending on GPU

## Memory Optimization Tips

If you run into memory issues:
1. Use lower resolution: `--resolution 480p`
2. Enable TF32 for faster computation: `--enable_tf32`
3. Enable low memory mode: `--low_memory_mode` (uses even smaller resolution)
4. The model automatically offloads to CPU during generation

### FP8 Model Support

The model file `Wan2_1-I2V-ATI-14B_fp8_e4m3fn.safetensors` is quantized to FP8 format, which significantly reduces memory usage from ~65GB (FP32) to ~16GB (FP8). The implementation:
- Automatically detects PyTorch FP8 support (requires PyTorch 2.1+)
- Keeps model weights in native FP8 format (torch.float8_e4m3fn) when supported
- Uses BFloat16 for activations and intermediate computations
- Falls back to BFloat16 conversion if FP8 is not supported

With FP8 support enabled:
- Model weights: ~16.4GB (1 byte per parameter)
- Total VRAM usage: ~20-24GB during inference

### Testing and Debugging

Test scripts are provided:
- `test_memory.py` - Check FP8 support and memory capabilities
- `test_model_load.py <checkpoint_dir>` - Test model loading and memory usage

## Troubleshooting

1. **Out of Memory**: The model requires significant VRAM. Try:
   - Using `--resolution 480p` for lower memory usage
   - Enable `--low_memory_mode` for aggressive memory optimization
   - Enable `--enable_tf32` for faster computation
   - The script already includes automatic model offloading

2. **Slow Generation**: This is normal for large diffusion models. Each of the 40 sampling steps requires a full forward pass through the 14B parameter model.

3. **Missing Checkpoints**: Ensure all required model files are in the checkpoint directory with correct names.

4. **CUDA/RTX 5090 Issues**: For newer GPUs like RTX 5090:
   - Ensure you have PyTorch 2.0+ with CUDA 12.1+ support
   - The model will automatically use compute-compatible dtypes

## License

This implementation follows the original WAN model license. Please refer to the official repository for licensing details.
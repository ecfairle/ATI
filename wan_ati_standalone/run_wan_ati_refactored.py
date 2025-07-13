#!/usr/bin/env python3
"""
Refactored WAN ATI (Animate Through Imagination) inference script.
This script runs image-to-video generation with motion trajectory control
using the refactored modular architecture.

Maintains compatibility with all flags from run_wan_ati.py
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from PIL import Image
import torch

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wan.image2video_refactored import WanATIRefactored
from wan.utils.motion import get_tracks_inference
from wan.utils.utils import cache_video
from wan.configs.wan_i2v_14B import i2v_14B

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)]
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate video from image using refactored WAN ATI model"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to directory containing model checkpoints"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="examples/images/bear.jpg",
        help="Path to input image (default: examples/images/bear.jpg)"
    )
    parser.add_argument(
        "--tracks",
        type=str,
        default="examples/tracks/bear.pth",
        help="Path to motion trajectory file (default: examples/tracks/bear.pth)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A brown bear lying in the shade beside a rock, resting on a bed of grass.",
        help="Text prompt describing the desired video"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path (default: wan_ati_output_TIMESTAMP.mp4)"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="720p",
        choices=["480p", "720p", "1080p"],
        help="Output resolution (default: 720p)"
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=40,
        help="Number of diffusion sampling steps (default: 40)"
    )
    parser.add_argument(
        "--guide_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale (default: 5.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on (default: cuda:0)"
    )
    parser.add_argument(
        "--enable_xformers",
        action="store_true",
        help="Enable xformers memory efficient attention"
    )
    parser.add_argument(
        "--enable_tf32",
        action="store_true",
        help="Enable TF32 for faster computation"
    )
    parser.add_argument(
        "--vae_cpu_offload",
        action="store_true",
        help="Offload VAE to CPU after encoding"
    )
    parser.add_argument(
        "--low_memory_mode",
        action="store_true",
        help="Enable aggressive memory optimizations (slower but uses less VRAM)"
    )
    
    # Additional flags for refactored version
    parser.add_argument(
        "--vae_only",
        action="store_true",
        help="Load only VAE for testing encoding/decoding"
    )
    parser.add_argument(
        "--preprocess_only",
        action="store_true",
        help="Run only preprocessing without loading models"
    )
    parser.add_argument(
        "--lazy_load",
        action="store_true",
        help="Load models on demand during generation"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Memory optimizations
    if args.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logging.info("Enabled TF32 for faster computation")
    
    # Set memory fraction to avoid fragmentation
    torch.cuda.set_per_process_memory_fraction(0.95)
    
    # Clear cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Set device
    device_id = int(args.device.split(':')[1]) if ':' in args.device else 0
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU (this will be very slow)")
    
    # Load configuration
    config = i2v_14B
    logging.info(f"Using configuration: WAN I2V 14B (Refactored)")
    
    # Initialize model based on mode
    logging.info(f"Loading WAN ATI model from {args.checkpoint_dir}")
    
    if args.vae_only:
        logging.info("VAE-only mode: Loading only VAE model")
        wan_ati = WanATIRefactored.create_with_vae_only(
            config=config,
            checkpoint_dir=args.checkpoint_dir,
            device_id=device_id,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=False,
            init_on_cpu=False
        )
    elif args.preprocess_only:
        logging.info("Preprocess-only mode: No models will be loaded")
        wan_ati = WanATIRefactored.create_for_testing(
            config=config,
            checkpoint_dir=args.checkpoint_dir,
            device_id=device_id,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=False,
            init_on_cpu=False
        )
    elif args.lazy_load:
        logging.info("Lazy loading mode: Models will be loaded on demand")
        wan_ati = WanATIRefactored(
            config=config,
            checkpoint_dir=args.checkpoint_dir,
            device_id=device_id,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=False,
            init_on_cpu=False
        )
    else:
        logging.info("Full pipeline mode: Loading all models")
        wan_ati = WanATIRefactored.create_full_pipeline(
            config=config,
            checkpoint_dir=args.checkpoint_dir,
            device_id=device_id,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=False,
            init_on_cpu=False
        )
    
    # Load input image
    logging.info(f"Loading image from {args.image}")
    img = Image.open(args.image).convert("RGB")
    width, height = img.size
    logging.info(f"Image size: {width}x{height}")
    
    # Load motion tracks
    logging.info(f"Loading motion tracks from {args.tracks}")
    tracks = get_tracks_inference(args.tracks, height, width)
    
    # Generate output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = ""
        if args.vae_only:
            mode_suffix = "_vae_only"
        elif args.preprocess_only:
            mode_suffix = "_preprocess_only"
        elif args.lazy_load:
            mode_suffix = "_lazy"
        args.output = f"wan_ati_output{mode_suffix}_{timestamp}.mp4"
    
    # Set max area based on resolution
    resolution_map = {
        "480p": 480 * 832,   # Lower resolution
        "720p": 720 * 1280,  # Default
        "1080p": 1080 * 1920 # High resolution (needs more VRAM)
    }
    max_area = resolution_map[args.resolution]
    
    # In low memory mode, use even smaller resolution
    if args.low_memory_mode:
        max_area = min(max_area, 480 * 640)
        logging.info("Low memory mode enabled - using reduced resolution")
    
    # Handle different modes
    if args.preprocess_only:
        # Just run preprocessing
        logging.info("Running preprocessing only...")
        preprocessed = wan_ati.preprocess_inputs(
            image=img,
            prompt=args.prompt,
            negative_prompt="",
            tracks=tracks,
            max_area=max_area,
            num_frames=81
        )
        
        logging.info("Preprocessing complete!")
        logging.info(f"  - Normalized image shape: {preprocessed['normalized_image'].shape}")
        logging.info(f"  - VAE input shape: {preprocessed['vae_input'].shape}")
        logging.info(f"  - Dimensions: {preprocessed['dimensions']}")
        logging.info(f"  - Max sequence length: {preprocessed['max_sequence_length']}")
        return
    
    if args.vae_only:
        # Test VAE encoding/decoding
        logging.info("Running VAE-only test...")
        preprocessed = wan_ati.preprocess_inputs(
            image=img,
            prompt=args.prompt,
            negative_prompt="",
            tracks=tracks,
            max_area=max_area,
            num_frames=81
        )
        
        # Encode through VAE
        latent = wan_ati.encode_with_vae(preprocessed)
        logging.info(f"VAE encoded latent shape: {latent.shape}")
        
        # Decode back (remove conditioning channels)
        decoded = wan_ati.vae.decode([latent[:16]])
        logging.info(f"VAE decoded shape: {decoded[0].shape}")
        
        # Save decoded result
        logging.info(f"Saving VAE test output to {args.output}")
        cache_video(
            tensor=decoded,
            save_file=args.output,
            fps=24,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )
        return
    
    # Full generation
    logging.info(f"Generating video with prompt: {args.prompt}")
    logging.info(f"Parameters: steps={args.sampling_steps}, guidance={args.guide_scale}, seed={args.seed}")
    
    # Log memory usage before generation
    if torch.cuda.is_available():
        logging.info(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        logging.info(f"GPU memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    
    video = wan_ati.generate(
        input_prompt=args.prompt,
        img=img,
        tracks=tracks,
        max_area=max_area,
        frame_num=81,         # 81 frames (must be 4n+1)
        shift=5.0,            # Noise schedule shift
        sample_solver='unipc',
        sampling_steps=args.sampling_steps,
        guide_scale=args.guide_scale,
        n_prompt="",          # Empty negative prompt
        seed=args.seed,
        offload_model=args.low_memory_mode or args.vae_cpu_offload  # Offload based on flags
    )
    
    # Save video
    logging.info(f"Saving video to {args.output}")
    cache_video(
        tensor=video[None],
        save_file=args.output,
        fps=24,               # 24 FPS output
        nrow=1,
        normalize=True,
        value_range=(-1, 1)
    )
    
    logging.info(f"Video generation complete! Output saved to {args.output}")
    logging.info(f"Video shape: {video.shape} (C, N, H, W)")
    
    # Log final memory usage
    if torch.cuda.is_available():
        logging.info(f"Final GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        logging.info(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")


if __name__ == "__main__":
    main()
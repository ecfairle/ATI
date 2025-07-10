#!/usr/bin/env python3
"""
Simplified WAN ATI (Animate Through Imagination) inference script.
This script runs image-to-video generation with motion trajectory control.
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

from wan import WanATI
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
        description="Generate video from image using WAN ATI model"
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
    logging.info(f"Using configuration: WAN I2V 14B")
    
    # Initialize model
    logging.info(f"Loading WAN ATI model from {args.checkpoint_dir}")
    wan_ati = WanATI(
        config=config,
        checkpoint_dir=args.checkpoint_dir,
        device_id=device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True
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
        args.output = f"wan_ati_output_{timestamp}.mp4"
    
    # Run generation
    logging.info(f"Generating video with prompt: {args.prompt}")
    logging.info(f"Parameters: steps={args.sampling_steps}, guidance={args.guide_scale}, seed={args.seed}")
    
    # Set max area based on resolution
    resolution_map = {
        "480p": 480 * 832,   # Lower resolution
        "720p": 720 * 1280,  # Default
        "1080p": 1080 * 1920 # High resolution (needs more VRAM)
    }
    max_area = resolution_map[args.resolution]
    
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
        offload_model=True    # Offload to CPU to save VRAM
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


if __name__ == "__main__":
    main()
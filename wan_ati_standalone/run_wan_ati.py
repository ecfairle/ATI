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
    return parser.parse_args()


def main():
    args = parse_args()
    
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
    
    video = wan_ati.generate(
        input_prompt=args.prompt,
        img=img,
        tracks=tracks,
        max_area=720 * 1280,  # 720p resolution
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
#!/usr/bin/env python3
"""
Validate inputs to WAN ATI model up to patch_motion step.
Only loads the VAE encoder, not the full model, CLIP or T5.
"""

import argparse
import os
import sys
import math
import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as TF

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wan.modules.vae import WanVAE
from wan.modules.motion_patch import patch_motion
from wan.utils.motion import get_tracks_inference


class InputValidator:
    """Validates inputs for WAN ATI model up to patch_motion step."""
    
    def __init__(self, device: str = "cuda", dtype=torch.bfloat16):
        self.device = torch.device(device)
        self.dtype = dtype
        self.vae = None
        
        # Model constraints - matching WanATI config
        self.vae_stride = (4, 8, 8)  # temporal, height, width
        self.patch_size = (1, 2, 2)
        self.sp_size = 1  # sequence packing size
        
    def load_vae_encoder(self, vae_checkpoint: str):
        """Load VAE following WanATI initialization."""
        logging.info(f"Loading VAE from {vae_checkpoint}")
        
        # Initialize VAE using WanVAE class from image2video.py
        self.vae = WanVAE(
            z_dim=16,
            vae_pth=vae_checkpoint,
            dtype=self.dtype,
            device=self.device
        )
        
        logging.info(f"VAE loaded successfully")
        
    def validate_image(self, image_path: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Validate and preprocess input image."""
        validation = {"status": "success", "issues": []}
        
        # Check file exists
        if not os.path.exists(image_path):
            validation["status"] = "error"
            validation["issues"].append(f"Image file not found: {image_path}")
            return None, validation
            
        try:
            # Load image
            img = Image.open(image_path).convert("RGB")
            validation["original_size"] = img.size
            
            # Check image dimensions
            if img.width < 64 or img.height < 64:
                validation["issues"].append(f"Image too small: {img.size}. Minimum size is 64x64")
                
            # Convert to tensor and normalize - matching image2video.py line 236
            img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
            
            validation["tensor_shape"] = list(img_tensor.shape)
            validation["tensor_dtype"] = str(img_tensor.dtype)
            validation["tensor_device"] = str(img_tensor.device)
            validation["tensor_range"] = [img_tensor.min().item(), img_tensor.max().item()]
            
            return img_tensor, validation
            
        except Exception as e:
            validation["status"] = "error"
            validation["issues"].append(f"Failed to load image: {str(e)}")
            return None, validation
            
    def validate_tracks(self, tracks_path: Optional[str], height: int, width: int) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Validate motion tracks using get_tracks_inference like in run_wan_ati.py."""
        validation = {"status": "success", "issues": []}
        
        if not tracks_path:
            validation["no_tracks"] = True
            return None, validation
            
        # Check file exists
        if not os.path.exists(tracks_path):
            validation["status"] = "error"
            validation["issues"].append(f"Tracks file not found: {tracks_path}")
            return None, validation
            
        try:
            # Load tracks using get_tracks_inference - matching run_wan_ati.py
            tracks = get_tracks_inference(tracks_path, height, width)
            tracks = tracks.to(self.device)[None]  # Add batch dimension like in image2video.py line 237
            
            validation["tracks_shape"] = list(tracks.shape)
            validation["tracks_dtype"] = str(tracks.dtype)
            
            # Validate tracks shape
            if len(tracks.shape) != 4:
                validation["issues"].append(f"Tracks should have 4 dimensions [B, T, N, 4], got {tracks.shape}")
            else:
                B, T, N, coords = tracks.shape
                if coords != 4:
                    validation["issues"].append(f"Tracks should have 4 coordinates per point, got {coords}")
                if T != 81:  # Expected frames
                    validation["issues"].append(f"Tracks frames ({T}) doesn't match expected 81 frames")
                    
            # Check coordinate ranges
            if tracks.numel() > 0:
                validation["coord_range"] = [tracks.min().item(), tracks.max().item()]
                
            return tracks, validation
            
        except Exception as e:
            validation["status"] = "error"
            validation["issues"].append(f"Failed to load tracks: {str(e)}")
            return None, validation
            
    def compute_latent_dimensions(self, img_shape: tuple, max_area: int, frame_num: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Compute latent dimensions following image2video.py lines 242-256."""
        validation = {"status": "success", "issues": []}
        
        # Check frame count constraint
        if (frame_num - 1) % 4 != 0:
            validation["issues"].append(f"frame_num must be 4n+1, got {frame_num}")
            
        F = frame_num
        h, w = img_shape[1:] if len(img_shape) == 3 else img_shape
        aspect_ratio = h / w
        
        # Compute latent dimensions - matching image2video.py lines 245-252
        # Compute latent dimensions - matching image2video.py lines 245-252
        # lat_h = round(
        #     np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
        #     self.patch_size[1] * self.patch_size[1])
        # lat_w = round(
        #     np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
        #     self.patch_size[2] * self.patch_size[2])
        # h = lat_h * self.vae_stride[1]
        # w = lat_w * self.vae_stride[2]

        lat_h = h // 8
        lat_w = w // 8
        
        # Compute max sequence length - matching image2video.py lines 254-256
        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
        
        dims = {
            "original_h": img_shape[1] if len(img_shape) == 3 else img_shape[0],
            "original_w": img_shape[2] if len(img_shape) == 3 else img_shape[1],
            "aspect_ratio": aspect_ratio,
            "h": h,
            "w": w,
            "lat_h": lat_h,
            "lat_w": lat_w,
            "max_seq_len": max_seq_len,
            "frame_num": F
        }
        
        validation["dimensions"] = dims
        
        return dims, validation
        
    def validate_vae_encoding(self, image: torch.Tensor, dims: dict) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Validate VAE encoding process following image2video.py lines 304-315."""
        validation = {"status": "success", "issues": []}
        
        if self.vae is None:
            validation["status"] = "error"
            validation["issues"].append("VAE not loaded")
            return None, validation
            
        try:
            h, w = dims["h"], dims["w"]
            F = dims["frame_num"]
            lat_h, lat_w = dims["lat_h"], dims["lat_w"]
            
            # Prepare video input - matching image2video.py lines 304-312
            # Resize image if needed
            if image.shape[-2] != h or image.shape[-1] != w:
                img_resized = torch.nn.functional.interpolate(
                    image[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1)
            else:
                img_resized = image[None].cpu().transpose(0, 1)
                
            video_input = torch.concat([
                img_resized,
                torch.zeros(3, F - 1, h, w)
            ], dim=1).to(self.device)
            
            validation["video_input_shape"] = list(video_input.shape)
            
            # Encode with VAE - matching image2video.py line 304
            with torch.no_grad():
                y = self.vae.encode([video_input])[0]
                
            validation["latent_shape"] = list(y.shape)
            validation["latent_dtype"] = str(y.dtype)
            validation["latent_device"] = str(y.device)
            validation["latent_stats"] = {
                "mean": y.mean().item(),
                "std": y.std().item(),
                "min": y.min().item(),
                "max": y.max().item()
            }
            
            # Create mask - matching image2video.py lines 272-280
            msk = torch.ones(1, 81, lat_h, lat_w, device=self.device, dtype=self.dtype)
            msk[:, 1:] = 0
            msk = torch.concat([
                torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
            ], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
            msk = msk.transpose(1, 2)[0]
            
            # Concatenate mask and latents - matching image2video.py line 314
            y = torch.concat([msk, y.to(self.dtype)])
            
            validation["final_shape"] = list(y.shape)
            validation["mask_shape"] = list(msk.shape)
                
            return y, validation
            
        except Exception as e:
            validation["status"] = "error"
            validation["issues"].append(f"VAE encoding failed: {str(e)}")
            return None, validation
            
    def validate_patch_motion(self, y: torch.Tensor, tracks: Optional[torch.Tensor]) -> Dict[str, Any]:
        """Validate patch_motion operation following image2video.py lines 317-319."""
        validation = {"status": "success", "issues": []}
        
        try:
            if tracks is None:
                validation["no_motion"] = True
                return validation
                
            # Apply patch_motion - matching image2video.py lines 317-318
            with torch.no_grad():
                y_before = y.clone()
                y_patched = patch_motion(tracks.type(y.dtype), y, training=False)
                
            validation["output_shape"] = list(y_patched.shape)
            validation["output_dtype"] = str(y_patched.dtype)
            validation["output_range"] = [y_patched.min().item(), y_patched.max().item()]
            
            # Check if motion was actually applied
            if torch.allclose(y_before, y_patched):
                validation["issues"].append("Motion patching had no effect on latents")
            else:
                diff = (y_patched - y_before).abs()
                validation["motion_stats"] = {
                    "mean_diff": diff.mean().item(),
                    "max_diff": diff.max().item(),
                    "affected_pixels": (diff > 1e-6).sum().item()
                }
                
        except Exception as e:
            validation["status"] = "error"
            validation["issues"].append(f"patch_motion failed: {str(e)}")
            
        return validation
        
    def validate_all(self, image_path: str, tracks_path: Optional[str], max_area: int = 576) -> Dict[str, Any]:
        """Run all validation steps following WanATI.generate() flow."""
        results = {
            "success": True,
            "steps": {}
        }
        
        # 1. Validate image
        logging.info("1. Validating input image...")
        image, img_validation = self.validate_image(image_path)
        results["steps"]["image"] = img_validation
        
        if img_validation["status"] == "error":
            results["success"] = False
            return results
            
        # 2. Validate dimensions
        logging.info("2. Computing latent dimensions...")
        dims, dim_validation = self.compute_latent_dimensions(image.shape, max_area, frame_num=81)
        results["steps"]["dimensions"] = dim_validation
        
        # 3. Validate tracks
        logging.info("3. Validating motion tracks...")
        tracks, track_validation = self.validate_tracks(tracks_path, dims["h"], dims["w"])
        results["steps"]["tracks"] = track_validation
        
        if track_validation["status"] == "error":
            results["success"] = False
            
        # 4. Validate VAE encoding
        logging.info("4. Validating VAE encoding...")
        y, vae_validation = self.validate_vae_encoding(image, dims)
        results["steps"]["vae_encoding"] = vae_validation
        
        if vae_validation["status"] == "error":
            results["success"] = False
            return results
            
        # 5. Validate patch_motion
        logging.info("5. Validating patch_motion...")
        motion_validation = self.validate_patch_motion(y, tracks)
        results["steps"]["patch_motion"] = motion_validation
        
        if motion_validation["status"] == "error":
            results["success"] = False
            
        # Collect all issues
        all_issues = []
        for step_name, step_result in results["steps"].items():
            if "issues" in step_result and step_result["issues"]:
                all_issues.extend([f"[{step_name}] {issue}" for issue in step_result["issues"]])
                
        results["all_issues"] = all_issues
        results["success"] = len(all_issues) == 0
        
        return results


def print_validation_results(results: Dict[str, Any]):
    """Pretty print validation results."""
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    
    if results["success"]:
        print("✓ All validation checks passed!")
    else:
        print("✗ Validation failed with issues:")
        
    if "all_issues" in results and results["all_issues"]:
        print("\nIssues found:")
        for issue in results["all_issues"]:
            print(f"  • {issue}")
            
    print("\nDetailed results by step:")
    for step_name, step_result in results["steps"].items():
        print(f"\n{step_name.upper()}:")
        
        # Skip printing issues again
        filtered_result = {k: v for k, v in step_result.items() if k != "issues"}
        
        for key, value in filtered_result.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Validate inputs for WAN ATI model")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--tracks", help="Path to motion tracks file")
    parser.add_argument("--vae-checkpoint", required=True, help="Path to VAE checkpoint")
    parser.add_argument("--max-area", type=int, default=576, help="Maximum area in latent space")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"],
                        help="Data type for computations")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    
    # Initialize validator
    validator = InputValidator(device=args.device, dtype=dtype)
    
    # Load VAE
    validator.load_vae_encoder(args.vae_checkpoint)
    
    # Run validation
    results = validator.validate_all(
        image_path=args.image,
        tracks_path=args.tracks,
        max_area=args.max_area
    )
    
    # Print results
    print_validation_results(results)
    
    # Exit with appropriate code
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
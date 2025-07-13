#!/usr/bin/env python3
"""
Validate inputs to WAN ATI model up to patch_motion step.
Only loads the VAE encoder, not the full model, CLIP or T5.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as TF

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wan.modules.vae import WanVAE
from wan.modules.motion_patch import patch_motion
from wan.modules.motion import get_tracks_inference, process_tracks


class InputValidator:
    """Validates inputs for WAN ATI model up to patch_motion step."""
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.vae = None
        
        # Model constraints
        self.max_frames = 81  # Must be 4n+1
        self.vae_stride = 8
        self.patch_size = 2
        self.max_area = 576  # Maximum area in latent space
        
    def load_vae_encoder(self, vae_checkpoint: str):
        """Load only the VAE encoder from checkpoint."""
        print(f"Loading VAE encoder from {vae_checkpoint}")
        
        # Initialize VAE
        self.vae = WanVAE(
            stride=(1, 8, 8),
            out_channels=3,
            latent_channels=32,
            base_channels=128,
            channel_multipliers=(1, 2, 4, 4),
            temporal_downsample_levels=(2,),
            temporal_downsample_factor=3,
            layers_per_block=2,
            spatio_temporal_resnet=True,
        ).to(self.device).eval()
        
        # Load checkpoint
        checkpoint = torch.load(vae_checkpoint, map_location=self.device)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
            
        # Filter only encoder weights
        encoder_state = {k: v for k, v in state_dict.items() if k.startswith("encoder.")}
        self.vae.load_state_dict(encoder_state, strict=False)
        
        print(f"VAE encoder loaded successfully")
        
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
                
            # Convert to tensor and normalize
            img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5)
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            validation["tensor_shape"] = list(img_tensor.shape)
            validation["tensor_dtype"] = str(img_tensor.dtype)
            validation["tensor_device"] = str(img_tensor.device)
            
            return img_tensor, validation
            
        except Exception as e:
            validation["status"] = "error"
            validation["issues"].append(f"Failed to load image: {str(e)}")
            return None, validation
            
    def validate_tracks(self, tracks_path: Optional[str], num_frames: int) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Validate motion tracks."""
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
            # Load tracks
            tracks, visibility = get_tracks_inference(tracks_path, num_frames)
            
            validation["tracks_shape"] = list(tracks.shape)
            validation["visibility_shape"] = list(visibility.shape)
            
            # Validate tracks shape
            if len(tracks.shape) != 3:
                validation["issues"].append(f"Tracks should have 3 dimensions [T, N, 4], got {tracks.shape}")
            else:
                T, N, coords = tracks.shape
                if coords != 4:
                    validation["issues"].append(f"Tracks should have 4 coordinates per point, got {coords}")
                if T != num_frames:
                    validation["issues"].append(f"Tracks frames ({T}) doesn't match num_frames ({num_frames})")
                    
            # Check coordinate ranges
            if tracks.numel() > 0:
                min_coords = tracks[..., :2].min().item()
                max_coords = tracks[..., :2].max().item()
                validation["coord_range"] = [min_coords, max_coords]
                
                if min_coords < 0 or max_coords > 1:
                    validation["issues"].append(f"Track coordinates should be in [0, 1], got [{min_coords}, {max_coords}]")
                    
            # Process tracks
            processed_tracks = process_tracks(tracks, visibility)
            processed_tracks = processed_tracks.to(self.device)
            
            validation["processed_shape"] = list(processed_tracks.shape)
            validation["processed_range"] = [
                processed_tracks[..., :2].min().item(),
                processed_tracks[..., :2].max().item()
            ]
            
            return processed_tracks, validation
            
        except Exception as e:
            validation["status"] = "error"
            validation["issues"].append(f"Failed to load tracks: {str(e)}")
            return None, validation
            
    def compute_latent_dimensions(self, width: int, height: int, num_frames: int) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """Compute and validate latent dimensions."""
        validation = {"status": "success", "issues": []}
        
        # Check frame count constraint
        if (num_frames - 1) % 4 != 0:
            validation["issues"].append(f"num_frames must be 4n+1, got {num_frames}")
            
        # Compute latent dimensions
        latent_width = width // self.vae_stride
        latent_height = height // self.vae_stride
        latent_frames = num_frames
        
        # Compute patch dimensions
        patch_width = latent_width // self.patch_size
        patch_height = latent_height // self.patch_size
        
        # Check area constraint
        area = patch_width * patch_height
        if area > self.max_area:
            validation["issues"].append(f"Patch area {area} exceeds max_area {self.max_area}")
            validation["issues"].append(f"Maximum resolution: {self.max_area * self.patch_size * self.vae_stride}px")
            
        dims = {
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "latent_width": latent_width,
            "latent_height": latent_height,
            "latent_frames": latent_frames,
            "patch_width": patch_width,
            "patch_height": patch_height,
            "patch_area": area
        }
        
        validation["dimensions"] = dims
        
        return dims, validation
        
    def validate_vae_encoding(self, image: torch.Tensor, num_frames: int) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Validate VAE encoding process."""
        validation = {"status": "success", "issues": []}
        
        if self.vae is None:
            validation["status"] = "error"
            validation["issues"].append("VAE not loaded")
            return None, validation
            
        try:
            B, C, H, W = image.shape
            
            # Expand image to video format [B, C, T, H, W]
            video = image.unsqueeze(2).expand(-1, -1, num_frames, -1, -1)
            validation["video_shape"] = list(video.shape)
            
            # Encode with VAE
            with torch.no_grad():
                latents = self.vae.encode(video)
                
            validation["latent_shape"] = list(latents.shape)
            validation["latent_dtype"] = str(latents.dtype)
            validation["latent_device"] = str(latents.device)
            validation["latent_stats"] = {
                "mean": latents.mean().item(),
                "std": latents.std().item(),
                "min": latents.min().item(),
                "max": latents.max().item()
            }
            
            # Check latent dimensions
            expected_h = H // self.vae_stride
            expected_w = W // self.vae_stride
            
            if latents.shape[-2] != expected_h or latents.shape[-1] != expected_w:
                validation["issues"].append(
                    f"Unexpected latent spatial dimensions: got {latents.shape[-2]}x{latents.shape[-1]}, "
                    f"expected {expected_h}x{expected_w}"
                )
                
            return latents, validation
            
        except Exception as e:
            validation["status"] = "error"
            validation["issues"].append(f"VAE encoding failed: {str(e)}")
            return None, validation
            
    def validate_patch_motion(self, latents: torch.Tensor, tracks: Optional[torch.Tensor]) -> Dict[str, Any]:
        """Validate patch_motion operation."""
        validation = {"status": "success", "issues": []}
        
        try:
            if tracks is None:
                validation["no_motion"] = True
                return validation
                
            # Apply patch_motion
            with torch.no_grad():
                patched_latents = patch_motion(
                    latents,
                    tracks,
                    patch_size=self.patch_size
                )
                
            validation["output_shape"] = list(patched_latents.shape)
            validation["output_dtype"] = str(patched_latents.dtype)
            
            # Check if motion was actually applied
            if torch.allclose(latents, patched_latents):
                validation["issues"].append("Motion patching had no effect on latents")
            else:
                diff = (patched_latents - latents).abs()
                validation["motion_stats"] = {
                    "mean_diff": diff.mean().item(),
                    "max_diff": diff.max().item(),
                    "affected_pixels": (diff > 1e-6).sum().item()
                }
                
        except Exception as e:
            validation["status"] = "error"
            validation["issues"].append(f"patch_motion failed: {str(e)}")
            
        return validation
        
    def validate_all(self, image_path: str, tracks_path: Optional[str], width: int, height: int, num_frames: int) -> Dict[str, Any]:
        """Run all validation steps."""
        results = {
            "success": True,
            "steps": {}
        }
        
        # 1. Validate image
        print("\n1. Validating input image...")
        image, img_validation = self.validate_image(image_path)
        results["steps"]["image"] = img_validation
        
        if img_validation["status"] == "error":
            results["success"] = False
            return results
            
        # 2. Validate dimensions
        print("\n2. Validating dimensions...")
        dims, dim_validation = self.compute_latent_dimensions(width, height, num_frames)
        results["steps"]["dimensions"] = dim_validation
        
        # 3. Validate tracks
        print("\n3. Validating motion tracks...")
        tracks, track_validation = self.validate_tracks(tracks_path, num_frames)
        results["steps"]["tracks"] = track_validation
        
        if track_validation["status"] == "error":
            results["success"] = False
            
        # 4. Validate VAE encoding
        print("\n4. Validating VAE encoding...")
        
        # Resize image if needed
        if image.shape[-2] != height or image.shape[-1] != width:
            image = F.interpolate(image, size=(height, width), mode='bilinear', align_corners=False)
            results["steps"]["image"]["resized"] = True
            results["steps"]["image"]["resized_shape"] = list(image.shape)
            
        latents, vae_validation = self.validate_vae_encoding(image, num_frames)
        results["steps"]["vae_encoding"] = vae_validation
        
        if vae_validation["status"] == "error":
            results["success"] = False
            return results
            
        # 5. Validate patch_motion
        print("\n5. Validating patch_motion...")
        motion_validation = self.validate_patch_motion(latents, tracks)
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
        
    if results["all_issues"]:
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
    parser.add_argument("--width", type=int, default=768, help="Output width")
    parser.add_argument("--height", type=int, default=512, help="Output height")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of frames (must be 4n+1)")
    parser.add_argument("--device", default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = InputValidator(device=args.device)
    
    # Load VAE
    validator.load_vae_encoder(args.vae_checkpoint)
    
    # Run validation
    results = validator.validate_all(
        image_path=args.image,
        tracks_path=args.tracks,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames
    )
    
    # Print results
    print_validation_results(results)
    
    # Exit with appropriate code
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
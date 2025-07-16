# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Preprocessing module for WAN ATI image-to-video model
# This module separates all preprocessing steps from model loading
# for better testing and modularity

import logging
import math
import numpy as np
import torch
import torchvision.transforms.functional as TF
from typing import Tuple, Optional, Dict, Any

from .modules.tokenizers import HuggingfaceTokenizer


class ImagePreprocessor:
    """Handles all image preprocessing operations"""
    
    @staticmethod
    def normalize_image(image: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Normalize image from [0, 1] to [-1, 1] range
        
        Args:
            image: PIL Image or tensor
            device: Target device for tensor
            
        Returns:
            Normalized image tensor
        """
        # Convert to tensor if needed and normalize
        if not isinstance(image, torch.Tensor):
            image = TF.to_tensor(image)
        
        # Normalize from [0, 1] to [-1, 1]
        normalized = image.sub_(0.5).div_(0.5).to(device)
        
        logging.info(f"[Image Preprocessing] Shape: {normalized.shape}, "
                    f"range: [{normalized.min():.3f}, {normalized.max():.3f}]")
        
        return normalized
    
    @staticmethod
    def prepare_for_vae(
        image: torch.Tensor, 
        target_height: int, 
        target_width: int,
        num_frames: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Prepare image tensor for VAE encoding
        
        Args:
            image: Normalized image tensor [C, H, W]
            target_height: Target height for VAE
            target_width: Target width for VAE
            num_frames: Total number of frames
            
        Returns:
            Video tensor with first frame as image, rest as zeros
        """
        # Resize image to target dimensions
        resized = torch.nn.functional.interpolate(
            image[None].cpu(), 
            size=(target_height, target_width), 
            mode='bicubic'
        ).transpose(0, 1)
        
        # Create video tensor with first frame as image, rest as zeros
        video_tensor = torch.concat([
            resized,
            torch.zeros(3, num_frames - 1, target_height, target_width)
        ], dim=1).to(device)
        
        return video_tensor


class TextPreprocessor:
    """Handles all text preprocessing operations"""
    
    def __init__(self, tokenizer_path: str, text_len: int = 256):
        """
        Initialize text preprocessor
        
        Args:
            tokenizer_path: Path to tokenizer model
            text_len: Maximum text length
        """
        self.text_len = text_len
        self.tokenizer = HuggingfaceTokenizer(
            tokenizer_path,
            max_length=text_len,
            clean='canonicalize'
        )
    
    def preprocess_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Preprocess text prompt for encoding
        
        Args:
            prompt: Input text prompt
            
        Returns:
            Dictionary with tokenized IDs and mask
        """
        # Tokenize with cleaning
        ids, mask = self.tokenizer([prompt], return_mask=True, add_special_tokens=True)
        
        return {
            'input_ids': ids,
            'attention_mask': mask,
            'text': prompt
        }


class MotionPreprocessor:
    """Handles motion track preprocessing"""
    
    @staticmethod
    def normalize_tracks(
        tracks: torch.Tensor, 
        width: int, 
        height: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Normalize motion tracks to [-1, 1] coordinate system
        
        Args:
            tracks: Raw motion tracks tensor with shape (T, N, 4)
                    where last dimension is [batch_idx, x, y, visibility]
            width: Video width
            height: Video height
            device: Target device
            
        Returns:
            Normalized tracks tensor with same shape
        """
        # Convert to device and add batch dimension
        tracks = tracks.to(device)[None]  # Add batch dimension: (1, T, N, 4)
        
        logging.info(f"[Motion Preprocessing] Input tracks shape: {tracks.shape}, "
                    f"range: [{tracks.min():.3f}, {tracks.max():.3f}]")
        
        return tracks


class DimensionCalculator:
    """Calculate dimensions for VAE and latent space"""
    
    @staticmethod
    def compute_vae_dimensions(
        image_height: int,
        image_width: int,
        max_area: int,
        vae_stride: Tuple[int, int, int],
        patch_size: Tuple[int, int, int]
    ) -> Dict[str, int]:
        """
        Compute target dimensions for VAE encoding
        
        Args:
            image_height: Original image height
            image_width: Original image width
            max_area: Maximum pixel area
            vae_stride: VAE stride parameters
            patch_size: Patch size parameters
            
        Returns:
            Dictionary with computed dimensions
        """
        # aspect_ratio = image_height / image_width
        
        # # Calculate latent dimensions
        # lat_h = round(
        #     np.sqrt(max_area * aspect_ratio) // vae_stride[1] //
        #     patch_size[1] * patch_size[1]
        # )
        # lat_w = round(
        #     np.sqrt(max_area / aspect_ratio) // vae_stride[2] //
        #     patch_size[2] * patch_size[2]
        # )
        
        # # Calculate actual dimensions
        # height = lat_h * vae_stride[1]
        # width = lat_w * vae_stride[2]
        
        height = image_height
        width = image_width
        lat_h = image_height // 8
        lat_w = image_width // 8
        aspect_ratio = image_height / image_width
        return {
            'height': height,
            'width': width,
            'latent_height': lat_h,
            'latent_width': lat_w,
            'aspect_ratio': aspect_ratio
        }
    
    @staticmethod
    def compute_max_sequence_length(
        num_frames: int,
        latent_height: int,
        latent_width: int,
        vae_stride: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        sequence_parallel_size: int = 1
    ) -> int:
        """
        Compute maximum sequence length for model
        
        Args:
            num_frames: Number of video frames
            latent_height: Latent space height
            latent_width: Latent space width
            vae_stride: VAE stride parameters
            patch_size: Patch size parameters
            sequence_parallel_size: Sequence parallelism size
            
        Returns:
            Maximum sequence length
        """
        max_seq_len = ((num_frames - 1) // vae_stride[0] + 1) * latent_height * latent_width // (
            patch_size[1] * patch_size[2]
        )
        
        # Align to sequence parallel size
        max_seq_len = int(math.ceil(max_seq_len / sequence_parallel_size)) * sequence_parallel_size
        
        return max_seq_len


class PreprocessingPipeline:
    """Main preprocessing pipeline that combines all preprocessing steps"""
    
    def __init__(
        self,
        tokenizer_path: str,
        text_len: int,
        vae_stride: Tuple[int, int, int],
        patch_size: Tuple[int, int, int]
    ):
        """
        Initialize preprocessing pipeline
        
        Args:
            tokenizer_path: Path to text tokenizer
            text_len: Maximum text length
            vae_stride: VAE stride parameters
            patch_size: Patch size parameters
        """
        self.image_preprocessor = ImagePreprocessor()
        self.text_preprocessor = TextPreprocessor(tokenizer_path, text_len)
        self.motion_preprocessor = MotionPreprocessor()
        self.dimension_calculator = DimensionCalculator()
        
        self.vae_stride = vae_stride
        self.patch_size = patch_size
    
    def preprocess_all(
        self,
        image,
        prompt: str,
        negative_prompt: str,
        tracks: torch.Tensor,
        max_area: int,
        num_frames: int,
        device: torch.device,
        sequence_parallel_size: int = 1
    ) -> Dict[str, Any]:
        """
        Run complete preprocessing pipeline
        
        Args:
            image: Input PIL image
            prompt: Text prompt
            negative_prompt: Negative text prompt
            tracks: Motion tracks tensor
            max_area: Maximum pixel area
            num_frames: Number of frames
            device: Target device
            sequence_parallel_size: Sequence parallelism size
            
        Returns:
            Dictionary with all preprocessed data
        """
        # 1. Normalize image
        normalized_image = self.image_preprocessor.normalize_image(image, device)
        
        # 2. Calculate dimensions
        h, w = normalized_image.shape[1:]
        dimensions = self.dimension_calculator.compute_vae_dimensions(
            h, w, max_area, self.vae_stride, self.patch_size
        )
        
        # 3. Prepare image for VAE
        vae_input = self.image_preprocessor.prepare_for_vae(
            normalized_image,
            dimensions['height'],
            dimensions['width'],
            num_frames,
            device
        )
        
        # 4. Preprocess text
        prompt_tokens = self.text_preprocessor.preprocess_prompt(prompt)
        negative_prompt_tokens = self.text_preprocessor.preprocess_prompt(negative_prompt)
        
        # 5. Prepare motion tracks (just convert to device and add batch dim)
        normalized_tracks = self.motion_preprocessor.normalize_tracks(
            tracks,
            dimensions['width'],
            dimensions['height'],
            device
        )
        
        # 6. Calculate max sequence length
        max_seq_len = self.dimension_calculator.compute_max_sequence_length(
            num_frames,
            dimensions['latent_height'],
            dimensions['latent_width'],
            self.vae_stride,
            self.patch_size,
            sequence_parallel_size
        )
        
        # Create conditioning mask
        msk = self._create_conditioning_mask(
            num_frames,
            dimensions['latent_height'],
            dimensions['latent_width'],
            device
        )
        
        return {
            'normalized_image': normalized_image,
            'vae_input': vae_input,
            'prompt_tokens': prompt_tokens,
            'negative_prompt_tokens': negative_prompt_tokens,
            'normalized_tracks': normalized_tracks,
            'dimensions': dimensions,
            'max_sequence_length': max_seq_len,
            'conditioning_mask': msk
        }
    
    def _create_conditioning_mask(
        self,
        num_frames: int,
        latent_height: int,
        latent_width: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create conditioning mask for first frame
        
        Args:
            num_frames: Number of frames
            latent_height: Latent space height
            latent_width: Latent space width
            device: Target device
            
        Returns:
            Conditioning mask tensor
        """
        # Create mask with first frame as 1, rest as 0
        msk = torch.ones(1, 81, latent_height, latent_width, device=device, dtype=torch.float32)
        msk[:, 1:] = 0
        
        # Reshape for VAE format
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), 
            msk[:, 1:]
        ], dim=1)
        
        msk = msk.view(1, msk.shape[1] // 4, 4, latent_height, latent_width)
        msk = msk.transpose(1, 2)[0]
        
        return msk
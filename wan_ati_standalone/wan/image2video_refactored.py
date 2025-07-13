# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Refactored WAN ATI image-to-video model with separated loading sequence
# This version separates VAE loading, preprocessing, and inference model loading
# for better testing and modularity

import gc
import logging
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.motion_patch import patch_motion
from .modules.clip import CLIPModel
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .preprocessing import PreprocessingPipeline
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanATIRefactored:
    """
    Refactored WAN ATI model with separated initialization sequence:
    1. VAE loading can happen independently
    2. All preprocessing can be done without loading inference models
    3. Inference models are loaded only when needed
    """
    
    def __init__(
        self,
        config,
        checkpoint_dir: str,
        device_id: int = 0,
        rank: int = 0,
        t5_fsdp: bool = False,
        dit_fsdp: bool = False,
        use_usp: bool = False,
        t5_cpu: bool = False,
        init_on_cpu: bool = True,
    ):
        """
        Initialize base configuration without loading any models
        
        Args:
            config: Model configuration object
            checkpoint_dir: Directory containing model checkpoints
            device_id: Target GPU device ID
            rank: Process rank for distributed training
            t5_fsdp: Enable FSDP sharding for T5
            dit_fsdp: Enable FSDP sharding for DiT
            use_usp: Enable USP distribution strategy
            t5_cpu: Place T5 on CPU
            init_on_cpu: Initialize transformer on CPU
        """
        # Store configuration
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device(f"cuda:{device_id}")
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu
        self.t5_fsdp = t5_fsdp
        self.dit_fsdp = dit_fsdp
        self.init_on_cpu = init_on_cpu
        
        # Model configuration
        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype
        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.sample_neg_prompt = config.sample_neg_prompt
        
        # Model instances (loaded on demand)
        self.vae: Optional[WanVAE] = None
        self.text_encoder: Optional[T5EncoderModel] = None
        self.clip: Optional[CLIPModel] = None
        self.model: Optional[WanModel] = None
        
        # Preprocessing pipeline
        self.preprocessing_pipeline = PreprocessingPipeline(
            tokenizer_path=config.t5_tokenizer,
            text_len=config.text_len,
            vae_stride=config.vae_stride,
            patch_size=config.patch_size
        )
        
        # Compute dtype (determined when loading DiT model)
        self.compute_dtype = None
        self.using_fp8 = False
        
        # Sequence parallel size
        self.sp_size = 1
        
        logging.info("WanATIRefactored initialized - no models loaded yet")
    
    # ========== Model Loading Methods ==========
    
    def load_vae(self) -> WanVAE:
        """
        Load only the VAE model
        
        Returns:
            Loaded VAE model instance
        """
        if self.vae is not None:
            logging.info("VAE already loaded")
            return self.vae
        
        logging.info(f"Loading VAE from {self.checkpoint_dir}")
        self.vae = WanVAE(
            vae_pth=os.path.join(self.checkpoint_dir, self.config.vae_checkpoint),
            device=self.device
        )
        
        logging.info("VAE loaded successfully")
        return self.vae
    
    def load_text_encoder(self) -> T5EncoderModel:
        """
        Load the T5 text encoder model
        
        Returns:
            Loaded T5 encoder instance
        """
        if self.text_encoder is not None:
            logging.info("Text encoder already loaded")
            return self.text_encoder
        
        logging.info(f"Loading T5 text encoder from {self.checkpoint_dir}")
        shard_fn = partial(shard_model, device_id=self.device.index) if self.t5_fsdp else None
        
        self.text_encoder = T5EncoderModel(
            text_len=self.config.text_len,
            dtype=self.config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(self.checkpoint_dir, self.config.t5_checkpoint),
            tokenizer_path=self.config.t5_tokenizer,
            shard_fn=shard_fn,
        )
        
        logging.info("T5 text encoder loaded successfully")
        return self.text_encoder
    
    def load_clip_model(self) -> CLIPModel:
        """
        Load the CLIP vision model
        
        Returns:
            Loaded CLIP model instance
        """
        if self.clip is not None:
            logging.info("CLIP model already loaded")
            return self.clip
        
        logging.info(f"Loading CLIP from {self.checkpoint_dir}")
        self.clip = CLIPModel(
            dtype=self.config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(self.checkpoint_dir, self.config.clip_checkpoint),
            tokenizer_path=self.config.clip_tokenizer
        )
        
        logging.info("CLIP model loaded successfully")
        return self.clip
    
    def load_diffusion_model(self) -> WanModel:
        """
        Load the main diffusion (DiT) model
        
        Returns:
            Loaded diffusion model instance
        """
        if self.model is not None:
            logging.info("Diffusion model already loaded")
            return self.model
        
        logging.info(f"Loading WanModel from {self.checkpoint_dir}")
        safetensors_path = os.path.join(
            self.checkpoint_dir, 
            'Wan2_1-I2V-ATI-14B_fp8_e4m3fn.safetensors'
        )
        
        # Create model config
        model_config = {
            '_class_name': 'WanModel',
            '_diffusers_version': '0.30.0',
            'model_type': 'i2v',
            'text_len': self.config.text_len,
            'in_dim': 36,  # 16 VAE + 20 conditioning channels
            'dim': self.config.dim,
            'ffn_dim': self.config.ffn_dim,
            'freq_dim': self.config.freq_dim,
            'out_dim': 16,
            'num_heads': self.config.num_heads,
            'num_layers': self.config.num_layers,
            'eps': self.config.eps
        }
        
        # Determine dtype for model loading
        if "fp8" in safetensors_path:
            fp8_supported = hasattr(torch, 'float8_e4m3fn')
            if fp8_supported:
                logging.info("FP8 model detected - will keep in native FP8 format")
                self.compute_dtype = torch.bfloat16
                keep_fp8 = True
                dtype_override = None
            else:
                logging.info("FP8 model detected but FP8 not supported - will convert to bfloat16")
                dtype_override = torch.bfloat16
                self.compute_dtype = torch.bfloat16
                keep_fp8 = False
        else:
            dtype_override = self.param_dtype
            self.compute_dtype = self.param_dtype
            keep_fp8 = False
        
        # Load model
        self.model = WanModel.from_single_file(
            safetensors_path,
            config=model_config,
            dtype_override=dtype_override,
            keep_fp8=keep_fp8
        )
        self.model.eval().requires_grad_(False)
        self.using_fp8 = keep_fp8
        
        # Apply USP if needed
        if self.use_usp:
            self._apply_usp_to_model()
        
        # Handle FSDP and device placement
        if self.t5_fsdp or self.dit_fsdp or self.use_usp:
            init_on_cpu = False
        else:
            init_on_cpu = self.init_on_cpu
        
        if dist.is_initialized():
            dist.barrier()
        
        if self.dit_fsdp:
            shard_fn = partial(shard_model, device_id=self.device.index)
            self.model = shard_fn(self.model)
        else:
            if not init_on_cpu:
                if self.using_fp8:
                    from .utils.fp8_model_loader import move_model_to_device_fp8
                    self.model = move_model_to_device_fp8(self.model, str(self.device))
                else:
                    self.model.to(self.device)
        
        logging.info("Diffusion model loaded successfully")
        return self.model
    
    def load_inference_models(self) -> Dict[str, Any]:
        """
        Load all inference models (T5, CLIP, DiT)
        
        Returns:
            Dictionary with loaded model references
        """
        models = {
            'text_encoder': self.load_text_encoder(),
            'clip': self.load_clip_model(),
            'diffusion': self.load_diffusion_model()
        }
        
        logging.info("All inference models loaded")
        return models
    
    def _apply_usp_to_model(self):
        """Apply USP (Unified Sequence Parallelism) to the model"""
        from xfuser.core.distributed import get_sequence_parallel_world_size
        from .distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward
        
        for block in self.model.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        
        self.model.forward = types.MethodType(usp_dit_forward, self.model)
        self.sp_size = get_sequence_parallel_world_size()
    
    # ========== Preprocessing Methods ==========
    
    def preprocess_inputs(
        self,
        image,
        prompt: str,
        negative_prompt: str,
        tracks: torch.Tensor,
        max_area: int = 720 * 1280,
        num_frames: int = 81,
    ) -> Dict[str, Any]:
        """
        Run all preprocessing without requiring model loading
        
        Args:
            image: Input PIL image
            prompt: Text prompt
            negative_prompt: Negative text prompt
            tracks: Motion tracks tensor
            max_area: Maximum pixel area
            num_frames: Number of frames
            
        Returns:
            Dictionary with preprocessed data
        """
        # Use preprocessing pipeline
        preprocessed = self.preprocessing_pipeline.preprocess_all(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            tracks=tracks,
            max_area=max_area,
            num_frames=num_frames,
            device=self.device,
            sequence_parallel_size=self.sp_size
        )
        
        return preprocessed
    
    # ========== VAE Encoding (requires only VAE) ==========
    
    def encode_with_vae(
        self,
        preprocessed_data: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Encode preprocessed image through VAE
        
        Args:
            preprocessed_data: Output from preprocess_inputs
            
        Returns:
            VAE encoded latent tensor
        """
        # Ensure VAE is loaded
        if self.vae is None:
            self.load_vae()
        
        # Encode through VAE
        vae_input = preprocessed_data['vae_input']
        latent = self.vae.encode([vae_input])[0]
        
        # Add conditioning mask
        mask = preprocessed_data['conditioning_mask']
        latent_with_mask = torch.concat([mask, latent.to(self.compute_dtype or torch.float32)])
        
        logging.info(f"[VAE Encode] Latent shape: {latent_with_mask.shape}, "
                    f"range: [{latent_with_mask.min():.3f}, {latent_with_mask.max():.3f}]")
        
        return latent_with_mask
    
    # ========== Text/Image Encoding (requires inference models) ==========
    
    def encode_text(
        self,
        preprocessed_data: Dict[str, Any]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Encode text prompts through T5
        
        Args:
            preprocessed_data: Output from preprocess_inputs
            
        Returns:
            Tuple of (prompt_embeddings, negative_prompt_embeddings)
        """
        # Ensure text encoder is loaded
        if self.text_encoder is None:
            self.load_text_encoder()
        
        # Get tokenized inputs
        prompt_tokens = preprocessed_data['prompt_tokens']
        negative_tokens = preprocessed_data['negative_prompt_tokens']
        
        # Move to device and encode
        device = torch.device('cpu') if self.t5_cpu else self.device
        
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
        
        # Encode prompts
        prompt_ids = prompt_tokens['input_ids'].to(device)
        prompt_mask = prompt_tokens['attention_mask'].to(device)
        context = self.text_encoder.model(prompt_ids, prompt_mask)
        
        # Encode negative prompts
        neg_ids = negative_tokens['input_ids'].to(device)
        neg_mask = negative_tokens['attention_mask'].to(device)
        context_null = self.text_encoder.model(neg_ids, neg_mask)
        
        # Move to target device if needed
        if self.t5_cpu:
            context = [c.to(self.device) for c in [context]]
            context_null = [c.to(self.device) for c in [context_null]]
        else:
            context = [context]
            context_null = [context_null]
        
        return context, context_null
    
    def encode_image_clip(
        self,
        preprocessed_data: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Encode image through CLIP
        
        Args:
            preprocessed_data: Output from preprocess_inputs
            
        Returns:
            CLIP image embeddings
        """
        # Ensure CLIP is loaded
        if self.clip is None:
            self.load_clip_model()
        
        # Get normalized image
        image = preprocessed_data['normalized_image']
        
        # Encode through CLIP
        self.clip.model.to(self.device)
        clip_features = self.clip.visual([image[:, None, :, :]])
        
        logging.info(f"[CLIP] Features shape: {clip_features.shape}, "
                    f"range: [{clip_features.min():.3f}, {clip_features.max():.3f}]")
        
        return clip_features
    
    # ========== Complete Generation Pipeline ==========
    
    def generate(
        self,
        input_prompt: str,
        img,
        tracks: torch.Tensor,
        max_area: int = 720 * 1280,
        frame_num: int = 81,
        shift: float = 5.0,
        sample_solver: str = 'unipc',
        sampling_steps: int = 40,
        guide_scale: float = 5.0,
        n_prompt: str = "",
        seed: int = -1,
        offload_model: bool = True
    ) -> torch.Tensor:
        """
        Generate video using the refactored pipeline
        
        This method demonstrates the new modular approach:
        1. Preprocessing happens first
        2. VAE encoding can happen independently
        3. Inference models are loaded only when needed
        """
        # Set negative prompt
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        
        # Step 1: Preprocessing (no models needed)
        logging.info("Step 1: Preprocessing inputs")
        preprocessed = self.preprocess_inputs(
            image=img,
            prompt=input_prompt,
            negative_prompt=n_prompt,
            tracks=tracks,
            max_area=max_area,
            num_frames=frame_num
        )
        
        # Step 2: VAE Encoding (only VAE needed)
        logging.info("Step 2: VAE encoding")
        vae_latent = self.encode_with_vae(preprocessed)
        
        # Apply motion patching
        with torch.no_grad():
            vae_latent = patch_motion(
                preprocessed['normalized_tracks'].type(vae_latent.dtype),
                vae_latent,
                training=False
            )
        
        # Step 3: Load inference models (T5, CLIP, DiT)
        logging.info("Step 3: Loading inference models")
        self.load_inference_models()
        
        # Step 4: Encode text and image
        logging.info("Step 4: Encoding text and image")
        text_embeddings, negative_embeddings = self.encode_text(preprocessed)
        clip_features = self.encode_image_clip(preprocessed)
        
        # Offload models if requested
        if offload_model:
            self.text_encoder.model.cpu()
            self.clip.model.cpu()
        
        # Step 5: Run diffusion sampling
        logging.info("Step 5: Running diffusion sampling")
        generated_latent = self._run_diffusion_sampling(
            vae_latent=vae_latent,
            text_embeddings=text_embeddings,
            negative_embeddings=negative_embeddings,
            clip_features=clip_features,
            preprocessed=preprocessed,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            seed=seed,
            offload_model=offload_model
        )
        
        # Step 6: Decode through VAE
        logging.info("Step 6: VAE decoding")
        if self.rank == 0:
            videos = self.vae.decode([generated_latent])
            return videos[0]
        
        return None
    
    def _run_diffusion_sampling(
        self,
        vae_latent: torch.Tensor,
        text_embeddings: List[torch.Tensor],
        negative_embeddings: List[torch.Tensor],
        clip_features: torch.Tensor,
        preprocessed: Dict[str, Any],
        shift: float,
        sample_solver: str,
        sampling_steps: int,
        guide_scale: float,
        seed: int,
        offload_model: bool
    ) -> torch.Tensor:
        """
        Run the diffusion sampling process
        """
        # Initialize noise
        dimensions = preprocessed['dimensions']
        F = preprocessed['vae_input'].shape[1]
        
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        
        noise = torch.randn(
            16, (F - 1) // 4 + 1,
            dimensions['latent_height'],
            dimensions['latent_width'],
            dtype=self.compute_dtype,
            generator=seed_g,
            device=self.device
        )
        
        # Prepare model arguments
        arg_c = {
            'context': [text_embeddings[0].to(self.compute_dtype)],
            'clip_fea': clip_features.to(self.compute_dtype),
            'seq_len': preprocessed['max_sequence_length'],
            'y': [vae_latent],
        }
        
        arg_null = {
            'context': [c.to(self.compute_dtype) for c in negative_embeddings],
            'clip_fea': clip_features.to(self.compute_dtype),
            'seq_len': preprocessed['max_sequence_length'],
            'y': [vae_latent],
        }
        
        # Clean up memory
        if offload_model:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # Move model to device
        if self.using_fp8:
            from .utils.fp8_model_loader import move_model_to_device_fp8
            self.model = move_model_to_device_fp8(self.model, str(self.device))
        else:
            self.model.to(self.device)
        
        # Initialize scheduler
        if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False
            )
            sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=self.device,
                sigmas=sampling_sigmas
            )
        else:
            raise NotImplementedError(f"Unsupported solver: {sample_solver}")
        
        # Run sampling loop
        @contextmanager
        def noop_no_sync():
            yield
        
        no_sync = getattr(self.model, 'no_sync', noop_no_sync)
        
        with amp.autocast(dtype=self.compute_dtype), torch.no_grad(), no_sync():
            latent = noise
            
            for t in tqdm(timesteps):
                latent_model_input = [latent.to(self.device)]
                timestep = torch.tensor([t], device=self.device)
                
                # Conditional prediction
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c
                )[0].to(torch.device('cpu') if offload_model else self.device)
                
                if offload_model:
                    torch.cuda.empty_cache()
                
                # Unconditional prediction
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null
                )[0].to(torch.device('cpu') if offload_model else self.device)
                
                if offload_model:
                    torch.cuda.empty_cache()
                
                # Classifier-free guidance
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
                
                # Update latent
                latent = latent.to(torch.device('cpu') if offload_model else self.device)
                
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g
                )[0]
                
                latent = temp_x0.squeeze(0)
        
        # Clean up
        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()
        
        return latent.to(self.device)
    
    # ========== Factory Methods ==========
    
    @classmethod
    def create_for_testing(cls, config, checkpoint_dir: str, **kwargs) -> 'WanATIRefactored':
        """
        Create instance optimized for testing (minimal loading)
        
        Args:
            config: Model configuration
            checkpoint_dir: Checkpoint directory
            **kwargs: Additional init parameters
            
        Returns:
            WanATIRefactored instance with no models loaded
        """
        instance = cls(config, checkpoint_dir, **kwargs)
        logging.info("Created testing instance - no models loaded")
        return instance
    
    @classmethod
    def create_with_vae_only(cls, config, checkpoint_dir: str, **kwargs) -> 'WanATIRefactored':
        """
        Create instance with only VAE loaded
        
        Args:
            config: Model configuration
            checkpoint_dir: Checkpoint directory
            **kwargs: Additional init parameters
            
        Returns:
            WanATIRefactored instance with VAE loaded
        """
        instance = cls(config, checkpoint_dir, **kwargs)
        instance.load_vae()
        return instance
    
    @classmethod
    def create_full_pipeline(cls, config, checkpoint_dir: str, **kwargs) -> 'WanATIRefactored':
        """
        Create instance with all models loaded
        
        Args:
            config: Model configuration
            checkpoint_dir: Checkpoint directory
            **kwargs: Additional init parameters
            
        Returns:
            WanATIRefactored instance with all models loaded
        """
        instance = cls(config, checkpoint_dir, **kwargs)
        instance.load_vae()
        instance.load_inference_models()
        return instance
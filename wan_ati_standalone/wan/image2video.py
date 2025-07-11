# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Copyright (c) 2024-2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.motion_patch import patch_motion
from .modules.clip import CLIPModel
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanATI:
    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=config.t5_tokenizer,  # HuggingFace model ID
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir,
                                         config.clip_checkpoint),
            tokenizer_path=config.clip_tokenizer)  # HuggingFace model ID

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        safetensors_path = os.path.join(checkpoint_dir, 'Wan2_1-I2V-ATI-14B_fp8_e4m3fn.safetensors')
        
        # Create model config from our config
        model_config = {
            '_class_name': 'WanModel',
            '_diffusers_version': '0.30.0',
            'model_type': 'i2v',
            'text_len': config.text_len,
            'in_dim': 36,  # 16 VAE + 20 conditioning channels
            'dim': config.dim,
            'ffn_dim': config.ffn_dim,
            'freq_dim': config.freq_dim,
            'out_dim': 16,
            'num_heads': config.num_heads,
            'num_layers': config.num_layers,
            'eps': config.eps
        }
        
        # Determine dtype for model loading
        if "fp8" in safetensors_path:
            # Check if we can use native FP8
            fp8_supported = hasattr(torch, 'float8_e4m3fn')
            if fp8_supported:
                logging.info("FP8 model detected - will keep in native FP8 format")
                # Keep FP8 weights, but use bfloat16 for computations
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
        
        # Load model with appropriate dtype conversion
        self.model = WanModel.from_single_file(
            safetensors_path, 
            config=model_config,
            dtype_override=dtype_override,
            keep_fp8=keep_fp8
        )
        self.model.eval().requires_grad_(False)
        
        # Store whether we're using FP8
        self.using_fp8 = keep_fp8

        if t5_fsdp or dit_fsdp or use_usp:
            init_on_cpu = False

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            if not init_on_cpu:
                if self.using_fp8:
                    # Import here to avoid circular dependency
                    from .utils.fp8_model_loader import move_model_to_device_fp8
                    self.model = move_model_to_device_fp8(self.model, str(self.device))
                else:
                    self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 img,
                 tracks,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
        tracks = tracks.to(self.device)[None]
        
        logging.info(f"[Input] Image shape: {img.shape}, range: [{img.min():.3f}, {img.max():.3f}]")
        logging.info(f"[Input] Tracks shape: {tracks.shape}, range: [{tracks.min():.3f}, {tracks.max():.3f}]")

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16, (F - 1) // 4 + 1,
            lat_h,
            lat_w,
            dtype=self.compute_dtype,
            generator=seed_g,
            device=self.device)
        
        logging.info(f"[Noise] Initial noise shape: {noise.shape}, range: [{noise.min():.3f}, {noise.max():.3f}]")
        logging.info(f"[Dimensions] Video: {F} frames, {h}x{w} pixels, Latent: {lat_h}x{lat_w}, max_seq_len: {max_seq_len}")

        msk = torch.ones(1, 81, lat_h, lat_w, device=self.device, dtype=self.compute_dtype)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
            dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            logging.info(f"[Text Encoder] Context shape: {context[0].shape}, range: [{context[0].min():.3f}, {context[0].max():.3f}]")
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        self.clip.model.to(self.device)
        clip_context = self.clip.visual([img[:, None, :, :]])
        logging.info(f"[CLIP] Context shape: {clip_context.shape}, range: [{clip_context.min():.3f}, {clip_context.max():.3f}]")
        if offload_model:
            self.clip.model.cpu()

        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, F - 1, h, w)
            ],
                dim=1).to(self.device)
        ])[0]
        logging.info(f"[VAE Encode] Latent shape before mask: {y.shape}, range: [{y.min():.3f}, {y.max():.3f}]")
        y = torch.concat([msk, y.to(self.compute_dtype)])
        logging.info(f"[VAE Encode] Latent shape after mask: {y.shape}, range: [{y.min():.3f}, {y.max():.3f}]")

        with torch.no_grad():
            y = patch_motion(tracks.type(y.dtype), y, training=False)
            logging.info(f"[Motion Patch] Y shape after patching: {y.shape}, range: [{y.min():.3f}, {y.max():.3f}]")

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.compute_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise

            # Log tensor sizes before creating args
            logging.info(f"[Memory Debug] Context shape: {context[0].shape}, dtype: {context[0].dtype}")
            logging.info(f"[Memory Debug] Clip context shape: {clip_context.shape}, dtype: {clip_context.dtype}")
            logging.info(f"[Memory Debug] Y shape: {y.shape}, dtype: {y.dtype}")
            logging.info(f"[Memory Debug] Max seq len: {max_seq_len}")
            
            arg_c = {
                'context': [context[0].to(self.compute_dtype)],
                'clip_fea': clip_context.to(self.compute_dtype),
                'seq_len': max_seq_len,
                'y': [y],
            }

            arg_null = {
                'context': [c.to(self.compute_dtype) for c in context_null],
                'clip_fea': clip_context.to(self.compute_dtype),
                'seq_len': max_seq_len,
                'y': [y],
            }

            if offload_model:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            # Force garbage collection before model loading
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            if self.using_fp8:
                # Import here to avoid circular dependency
                from .utils.fp8_model_loader import move_model_to_device_fp8
                self.model = move_model_to_device_fp8(self.model, str(self.device))
            else:
                self.model.to(self.device)
            logging.info(f"[Sampling] Starting diffusion with {len(timesteps)} steps, solver: {sample_solver}")
            logging.info(f"[Memory] Before sampling: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f}GB reserved")
            
            for step_idx, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)
                
                # Log memory before model forward
                if step_idx == 0:
                    logging.info(f"[Memory] Before first model forward: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated")

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0].to(
                        torch.device('cpu') if offload_model else self.device)
                if offload_model:
                    torch.cuda.empty_cache()
                    
                # Log memory after first forward
                if step_idx == 0:
                    logging.info(f"[Memory] After first model forward: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated")
                    
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0].to(
                        torch.device('cpu') if offload_model else self.device)
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)
                
                if step_idx % 10 == 0 or step_idx == len(timesteps) - 1:
                    logging.info(f"[Step {step_idx}/{len(timesteps)-1}] t={t:.3f}, noise_pred range: [{noise_pred.min():.3f}, {noise_pred.max():.3f}], latent range: [{latent.min():.3f}, {latent.max():.3f}]")
                    logging.info(f"[Memory] {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated")

                latent = latent.to(
                    torch.device('cpu') if offload_model else self.device)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

                x0 = [latent.to(self.device)]
                del latent_model_input, timestep

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                logging.info(f"[VAE Decode] Input latent shape: {x0[0].shape}, range: [{x0[0].min():.3f}, {x0[0].max():.3f}]")
                videos = self.vae.decode(x0)
                logging.info(f"[Output] Video shape: {videos[0].shape}, range: [{videos[0].min():.3f}, {videos[0].max():.3f}]")

        del noise, latent
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

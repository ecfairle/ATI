# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------ Wan I2V 14B ------------------------#

i2v_14B = EasyDict(__name__='Config: Wan I2V 14B')
i2v_14B.update(wan_shared_cfg)
i2v_14B.sample_neg_prompt = "镜头晃动，" + i2v_14B.sample_neg_prompt

i2v_14B.t5_checkpoint = 'umt5-xxl-enc-bf16.safetensors'
i2v_14B.t5_tokenizer = 'google/umt5-xxl'

# clip
i2v_14B.clip_model = 'clip_xlm_roberta_vit_h_14'
i2v_14B.clip_dtype = torch.float16
i2v_14B.clip_checkpoint = 'clip_vision_h.safetensors'
i2v_14B.clip_tokenizer = 'xlm-roberta-large'

# vae
i2v_14B.vae_checkpoint = 'wan_2.1_vae.safetensors'
i2v_14B.vae_stride = (4, 8, 8)

# transformer
i2v_14B.patch_size = (1, 2, 2)
i2v_14B.dim = 5120
i2v_14B.ffn_dim = 13824
i2v_14B.freq_dim = 256
i2v_14B.num_heads = 40
i2v_14B.num_layers = 40
i2v_14B.window_size = (-1, -1)
i2v_14B.qk_norm = True
i2v_14B.cross_attn_norm = True
i2v_14B.eps = 1e-6

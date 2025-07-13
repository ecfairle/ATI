#!/usr/bin/env python3
# Example usage of the refactored WAN ATI model
# Demonstrates different initialization sequences for testing

import torch
from PIL import Image
from easydict import EasyDict

from wan.image2video_refactored import WanATIRefactored


def example_vae_only_testing():
    """
    Example: Test VAE encoding/decoding without loading other models
    """
    print("=== Example 1: VAE-only testing ===")
    
    # Mock configuration
    config = EasyDict({
        'vae_checkpoint': 'vae.safetensors',
        'vae_stride': (4, 8, 8),
        'patch_size': (1, 2, 2),
        'text_len': 256,
        't5_tokenizer': 'google/t5-v1_1-xxl',
        # Other config values...
    })
    
    # Create instance with only VAE
    model = WanATIRefactored.create_with_vae_only(
        config=config,
        checkpoint_dir='./checkpoints'
    )
    
    # Load test image
    image = Image.open('test_image.jpg')
    tracks = torch.randn(81, 10, 2)  # Mock tracks
    
    # Run preprocessing
    preprocessed = model.preprocess_inputs(
        image=image,
        prompt="A test prompt",
        negative_prompt="",
        tracks=tracks
    )
    
    # Encode through VAE
    latent = model.encode_with_vae(preprocessed)
    print(f"VAE latent shape: {latent.shape}")
    
    # Decode back
    decoded = model.vae.decode([latent[:16]])  # Remove conditioning channels
    print(f"Decoded shape: {decoded[0].shape}")
    
    print("VAE testing complete!\n")


def example_preprocessing_only():
    """
    Example: Test preprocessing without loading any models
    """
    print("=== Example 2: Preprocessing-only testing ===")
    
    # Mock configuration
    config = EasyDict({
        'vae_stride': (4, 8, 8),
        'patch_size': (1, 2, 2),
        'text_len': 256,
        't5_tokenizer': 'google/t5-v1_1-xxl',
        # Other config values...
    })
    
    # Create testing instance (no models loaded)
    model = WanATIRefactored.create_for_testing(
        config=config,
        checkpoint_dir='./checkpoints'
    )
    
    # Load test data
    image = Image.open('test_image.jpg')
    tracks = torch.randn(81, 10, 2)
    
    # Run preprocessing
    preprocessed = model.preprocess_inputs(
        image=image,
        prompt="A beautiful sunset over mountains",
        negative_prompt="blurry, low quality",
        tracks=tracks,
        max_area=720*1280,
        num_frames=81
    )
    
    # Check preprocessed outputs
    print(f"Normalized image shape: {preprocessed['normalized_image'].shape}")
    print(f"VAE input shape: {preprocessed['vae_input'].shape}")
    print(f"Computed dimensions: {preprocessed['dimensions']}")
    print(f"Max sequence length: {preprocessed['max_sequence_length']}")
    
    print("Preprocessing testing complete!\n")


def example_lazy_loading():
    """
    Example: Demonstrate lazy loading of models
    """
    print("=== Example 3: Lazy loading demonstration ===")
    
    # Mock configuration
    config = EasyDict({
        'vae_checkpoint': 'vae.safetensors',
        't5_checkpoint': 't5.safetensors',
        'clip_checkpoint': 'clip.safetensors',
        'vae_stride': (4, 8, 8),
        'patch_size': (1, 2, 2),
        'text_len': 256,
        't5_tokenizer': 'google/t5-v1_1-xxl',
        'clip_tokenizer': 'openai/clip-vit-large-patch14',
        't5_dtype': torch.float32,
        'clip_dtype': torch.float32,
        # Other config values...
    })
    
    # Create empty instance
    model = WanATIRefactored(
        config=config,
        checkpoint_dir='./checkpoints'
    )
    
    print("Initial state - no models loaded:")
    print(f"  VAE loaded: {model.vae is not None}")
    print(f"  Text encoder loaded: {model.text_encoder is not None}")
    print(f"  CLIP loaded: {model.clip is not None}")
    print(f"  Diffusion model loaded: {model.model is not None}")
    
    # Load models on demand
    print("\nLoading VAE...")
    model.load_vae()
    print(f"  VAE loaded: {model.vae is not None}")
    
    print("\nLoading text encoder...")
    model.load_text_encoder()
    print(f"  Text encoder loaded: {model.text_encoder is not None}")
    
    print("\nLoading remaining models...")
    model.load_inference_models()
    print(f"  All models loaded!")
    
    print("\nLazy loading demonstration complete!\n")


def example_modular_pipeline():
    """
    Example: Use the modular pipeline for custom workflows
    """
    print("=== Example 4: Modular pipeline usage ===")
    
    # Mock configuration
    config = EasyDict({
        'vae_checkpoint': 'vae.safetensors',
        'vae_stride': (4, 8, 8),
        'patch_size': (1, 2, 2),
        'text_len': 256,
        't5_tokenizer': 'google/t5-v1_1-xxl',
        't5_checkpoint': 't5.safetensors',
        'clip_checkpoint': 'clip.safetensors',
        'clip_tokenizer': 'openai/clip-vit-large-patch14',
        't5_dtype': torch.float32,
        'clip_dtype': torch.float32,
        # Other config values...
    })
    
    model = WanATIRefactored(
        config=config,
        checkpoint_dir='./checkpoints'
    )
    
    # Step 1: Preprocess (no models needed)
    image = Image.open('test_image.jpg')
    tracks = torch.randn(81, 10, 2)
    
    preprocessed = model.preprocess_inputs(
        image=image,
        prompt="A serene landscape",
        negative_prompt="",
        tracks=tracks
    )
    print("Step 1: Preprocessing complete")
    
    # Step 2: VAE encoding (only VAE needed)
    model.load_vae()
    vae_latent = model.encode_with_vae(preprocessed)
    print(f"Step 2: VAE encoding complete - latent shape: {vae_latent.shape}")
    
    # Step 3: Text encoding (only text encoder needed)
    model.load_text_encoder()
    text_emb, neg_emb = model.encode_text(preprocessed)
    print(f"Step 3: Text encoding complete - embedding shape: {text_emb[0].shape}")
    
    # Step 4: Image CLIP encoding (only CLIP needed)
    model.load_clip_model()
    clip_features = model.encode_image_clip(preprocessed)
    print(f"Step 4: CLIP encoding complete - features shape: {clip_features.shape}")
    
    print("\nModular pipeline demonstration complete!")
    print("Each step loaded only the models it needed.\n")


if __name__ == "__main__":
    # Note: These are examples showing the API usage
    # They won't run without actual model files and test images
    
    print("Refactored WAN ATI Usage Examples")
    print("=" * 50)
    print("\nThese examples demonstrate the new modular architecture:")
    print("1. VAE-only testing for encoding/decoding")
    print("2. Preprocessing without any model loading") 
    print("3. Lazy loading of models on demand")
    print("4. Modular pipeline with step-by-step execution")
    print("\nThe refactored code allows:")
    print("- Testing individual components in isolation")
    print("- Reduced memory usage during development")
    print("- Better separation of concerns")
    print("- Easier mocking and unit testing")
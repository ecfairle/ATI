# WAN ATI Model Refactoring Documentation

## Overview

This refactoring separates the WAN ATI image-to-video model's initialization sequence into three distinct phases:

1. **VAE Loading** - Load only the VAE model for encoding/decoding
2. **Preprocessing** - All preprocessing steps without model dependencies  
3. **Inference Model Loading** - Load T5, CLIP, and DiT models after preprocessing

## Key Changes

### New Files Created

1. **`preprocessing.py`** - Modular preprocessing components:
   - `ImagePreprocessor` - Image normalization and preparation
   - `TextPreprocessor` - Text cleaning and tokenization
   - `MotionPreprocessor` - Motion track normalization
   - `DimensionCalculator` - Compute VAE and latent dimensions
   - `PreprocessingPipeline` - Orchestrates all preprocessing

2. **`image2video_refactored.py`** - Refactored main class:
   - `WanATIRefactored` - New modular architecture
   - Separate loading methods for each model
   - Factory methods for different use cases

3. **`example_refactored_usage.py`** - Usage examples demonstrating the new API

## Architecture Improvements

### Before (Original `image2video.py`)
```python
class WanATI:
    def __init__(self, ...):
        # All models loaded immediately:
        self.text_encoder = T5EncoderModel(...)  # Loaded
        self.vae = WanVAE(...)                   # Loaded
        self.clip = CLIPModel(...)               # Loaded
        self.model = WanModel.from_single_file(...)  # Loaded
        
    def generate(self, ...):
        # Preprocessing mixed with model usage
        img = TF.to_tensor(img).sub_(0.5).div_(0.5)
        # ... rest of generation
```

### After (Refactored)
```python
class WanATIRefactored:
    def __init__(self, ...):
        # Only configuration stored, no models loaded
        self.vae = None
        self.text_encoder = None
        self.clip = None
        self.model = None
        
    def load_vae(self):
        # Load VAE independently
        
    def preprocess_inputs(self, ...):
        # All preprocessing without models
        
    def load_inference_models(self):
        # Load T5, CLIP, DiT when needed
```

## Usage Patterns

### 1. Test VAE Only
```python
# Create instance with just VAE
model = WanATIRefactored.create_with_vae_only(config, checkpoint_dir)

# Preprocess
preprocessed = model.preprocess_inputs(image, prompt, neg_prompt, tracks)

# Encode/decode through VAE
latent = model.encode_with_vae(preprocessed)
decoded = model.vae.decode([latent])
```

### 2. Test Preprocessing Only
```python
# Create testing instance (no models)
model = WanATIRefactored.create_for_testing(config, checkpoint_dir)

# Run preprocessing
preprocessed = model.preprocess_inputs(image, prompt, neg_prompt, tracks)

# Access preprocessed data
print(preprocessed['normalized_image'])
print(preprocessed['dimensions'])
```

### 3. Lazy Loading
```python
model = WanATIRefactored(config, checkpoint_dir)

# Load only what you need, when you need it
model.load_vae()  # Just VAE
model.load_text_encoder()  # Just T5
model.load_clip_model()  # Just CLIP
model.load_diffusion_model()  # Just DiT
```

### 4. Full Pipeline (Compatible with Original)
```python
# Create with all models
model = WanATIRefactored.create_full_pipeline(config, checkpoint_dir)

# Use generate() as before
video = model.generate(prompt, image, tracks, ...)
```

## Benefits

1. **Better Testing**
   - Test VAE encoding/decoding in isolation
   - Test preprocessing without loading 14B+ parameter models
   - Mock individual components easily

2. **Memory Efficiency**
   - Load only required models
   - Reduced memory footprint during development
   - Faster iteration cycles

3. **Cleaner Architecture**
   - Clear separation of concerns
   - Preprocessing logic isolated from model logic
   - Each component has single responsibility

4. **Backward Compatibility**
   - `generate()` method maintains same interface
   - Can be used as drop-in replacement with full pipeline

## Preprocessing Pipeline Details

The preprocessing is now organized into clear steps:

1. **Image Preprocessing**
   - Normalize to [-1, 1] range
   - Resize for VAE input
   - Create video tensor format

2. **Text Preprocessing**
   - Clean text (canonicalize, remove special chars)
   - Tokenize with T5 tokenizer
   - Create attention masks

3. **Motion Preprocessing**
   - Normalize tracks to [-1, 1] coordinate system
   - Handle aspect ratio correctly
   - Prepare for motion patching

4. **Dimension Calculation**
   - Compute VAE output dimensions
   - Calculate latent space dimensions
   - Determine max sequence length

## Migration Guide

To migrate existing code:

1. Replace `from wan.image2video import WanATI` with `from wan.image2video_refactored import WanATIRefactored`

2. For full compatibility, use factory method:
   ```python
   # Old
   model = WanATI(config, checkpoint_dir, ...)
   
   # New (full compatibility)
   model = WanATIRefactored.create_full_pipeline(config, checkpoint_dir, ...)
   ```

3. For testing workflows, use appropriate factory:
   ```python
   # VAE testing
   model = WanATIRefactored.create_with_vae_only(config, checkpoint_dir)
   
   # Preprocessing testing
   model = WanATIRefactored.create_for_testing(config, checkpoint_dir)
   ```

## Testing Recommendations

1. **Unit Tests for Preprocessing**
   ```python
   def test_image_preprocessing():
       preprocessor = ImagePreprocessor()
       normalized = preprocessor.normalize_image(test_image, device)
       assert normalized.min() >= -1.0
       assert normalized.max() <= 1.0
   ```

2. **Integration Tests for VAE**
   ```python
   def test_vae_encode_decode():
       model = WanATIRefactored.create_with_vae_only(config, checkpoint_dir)
       # Test encoding and decoding
   ```

3. **Mock Testing**
   ```python
   def test_generate_with_mocks():
       model = WanATIRefactored(config, checkpoint_dir)
       model.vae = MockVAE()
       model.text_encoder = MockT5()
       # Test generation logic without loading real models
   ```

## Future Enhancements

1. **Async Model Loading** - Load models in background while preprocessing
2. **Model Caching** - Cache loaded models across instances
3. **Preprocessing Cache** - Cache preprocessed results for repeated inputs
4. **Streaming Pipeline** - Process video frames in streaming fashion

## Summary

This refactoring provides a more modular, testable, and efficient architecture while maintaining backward compatibility. The separation of VAE loading, preprocessing, and inference model loading enables better testing workflows and reduces development friction when working with large models.
# BirdNET-CoreML

Convert BirdNET models (Keras/H5 format) to Apple Core ML format (.mlpackage) for on-device inference on iOS/macOS devices.

## Overview

This repository provides tools to convert BirdNET's bird detection models to Apple's Core ML format, enabling:

1. **Audio Model Conversion**: Convert the main BirdNET acoustic model that identifies bird species from audio
2. **Metadata Model Conversion**: Convert the location/time-based model that filters species by geographic occurrence
3. **Model Verification**: Tools to validate that converted models produce identical results

## Quick Setup

Use the automated setup script:

```bash
cd BirdNET-CoreML
./setup_environment.sh
source venv/bin/activate
```

This creates a clean virtual environment with all required dependencies.

## Converting Models

### Audio Model Conversion

The main conversion script handles both Keras/H5 models and TensorFlow SavedModel directories:

```bash
# Convert the included Keras/H5 model
python coreml_export/convert_keras_to_coreml.py \
  --in_path coreml_export/input/audio-model.h5 \
  --out_path coreml_export/output/audio-model.mlpackage

# Convert from a SavedModel directory
python coreml_export/convert_keras_to_coreml.py \
  --in_path path/to/savedmodel/ \
  --out_path output/model.mlpackage
```

**Options:**
- `--target ios15` - Minimum deployment target (default: ios15)
- `--keep_fp32` - Keep 32-bit precision (not recommended - models are already optimized for FP16)
- `--melspec_layer_file` - Specify MelSpecLayerSimple implementation (default: MelSpecLayerSimple_fixed.py for CoreML compatibility)

### Metadata Model Conversion

BirdNET uses a metadata model to filter predictions based on location and time of year:

```bash
# Convert the metadata model
python coreml_export/convert_meta_model_to_coreml.py \
  --input coreml_export/input/meta-model.h5 \
  --output coreml_export/output/metadata-model.mlpackage
```

## Using the Converted Models

### Audio Predictions

The converted audio model expects:
- **Input**: 3-second mono audio @ 48 kHz (144,000 samples)
- **Output**: Probability scores for 6,522 bird species

### Location-Based Filtering

Use the metadata model to filter predictions by location:

```python
from coreml_export.meta_utils import (
    get_species_priors, 
    filter_by_location,
    load_coreml_meta_model
)

# Load models
audio_model = coremltools.models.MLModel("audio-model.mlpackage")
meta_model = load_coreml_meta_model("metadata-model.mlpackage")

# Get audio predictions
audio_scores = audio_model.predict({"input": audio_data})

# Filter by location (latitude, longitude, week_of_year)
filtered_scores, filtered_labels = filter_by_location(
    audio_scores, species_labels, 
    latitude=40.7128, longitude=-74.0060, week=12, 
    meta_model=meta_model
)
```

## Model Verification

The repository includes verification tools:

### Compare Model Outputs

Compare predictions between different model formats:

```bash
# Compare Keras vs CoreML on test audio files
python verification/compare_model_predictions.py \
  --model1 coreml_export/input/audio-model.h5 \
  --model2 coreml_export/output/audio-model.mlpackage \
  --audio_dir verification/bird_sounds \
  --output_csv comparison_results.csv
```

### Verify Metadata Model

Test the metadata model with 17 geographic test cases:

```bash
python verification/verify_meta_models.py
```

## Project Structure

```
BirdNET-CoreML/
├── coreml_export/              # Main conversion scripts
│   ├── convert_keras_to_coreml.py    # Audio model converter
│   ├── convert_meta_model_to_coreml.py # Metadata model converter
│   ├── meta_utils.py           # Location filtering utilities
│   ├── input/                  # Input models and resources
│   │   ├── audio-model.h5      # Pre-trained audio model
│   │   ├── meta-model.h5       # Pre-trained metadata model
│   │   └── labels/             # Species labels in multiple languages
│   └── output/                 # Converted CoreML models
├── verification/               # Model validation tools
│   ├── compare_model_predictions.py  # Compare outputs across formats
│   ├── verify_meta_models.py   # Test metadata model
│   └── bird_sounds/            # Test audio samples
├── requirements.txt            # Python dependencies
├── setup_environment.sh        # Automated setup script
└── deprecated/                 # Legacy scripts no longer needed
    └── custom_layers.py        # Legacy SimpleSpecLayer (no longer used)
```

## Technical Details

### Model Architecture
- **Audio Model**: ResNet-based architecture with custom spectrogram preprocessing
- **Input**: Raw audio waveform (144,000 samples @ 48kHz)
- **Custom Layers**: 
- `MelSpecLayerSimple_fixed.py`: Modified mel-spectrogram layer that avoids CoreML-incompatible operations
- **Output**: 6,522 bird species probabilities

### Why MelSpecLayerSimple_fixed.py?
The original MelSpecLayerSimple layer used TensorFlow's `tf.abs()` operation on complex spectrograms, which isn't supported by CoreML. The fixed version manually computes the magnitude spectrum using `sqrt(real² + imag²)`, making it compatible with CoreML conversion while producing identical results.

### Model Precision
The models use FP16 (16-bit floating point) precision by default, which provides:
- ~50MB model size (vs ~100MB for FP32)
- Faster inference on Apple Neural Engine
- Negligible impact on accuracy

Initial experiments with FP32 showed no meaningful accuracy improvements, so FP16 is recommended.

### Supported Species
The model identifies 6,522 bird species globally, with labels available in 27 languages.

## Troubleshooting

### Import Errors
If you encounter import errors:
1. Delete your virtual environment
2. Run `./setup_environment.sh` to create a fresh environment

### TensorFlow Version Warning
You may see: `TensorFlow version 2.15.0 has not been tested with coremltools`

This warning can be safely ignored - the conversion has been thoroughly tested.

### Custom Layer Issues
The converter automatically handles BirdNET's custom layers. If you encounter issues:
- Ensure the correct MelSpecLayerSimple implementation is used (typically `MelSpecLayerSimple_fixed.py`)
- The fixed version avoids CoreML-incompatible operations while maintaining identical functionality
- The SimpleSpecLayer from `custom_layers.py` existed in early versions of BirdNET but is no longer used in current models

## Additional Notes on SimpleSpecLayer vs MelSpecLayerSimple

SimpleSpecLayer (deprecated):
- Creates a basic linear spectrogram (not mel-scaled)
- Uses STFT to get frequency bins linearly spaced
- Output shape: (257, 384) - likely 257 frequency bins
- Simple magnitude scaling and normalization
- No frequency warping to match human perception

MelSpecLayerSimple:
- Creates a mel-spectrogram (perceptually-scaled frequencies)
- Uses mel filterbank to convert linear frequencies to mel scale
- Includes frequency range parameters (fmin, fmax)
- More sophisticated preprocessing (normalization to [-1, 1] range)
- Better suited for audio ML tasks involving human-relevant sounds

Why MelSpecLayerSimple_fixed.py was needed

The key issue is on line 51 of the fixed version:

# Convert complex to magnitude spectrum manually (avoids ComplexAbs
  op)
real = tf.math.real(complex_spec)
imag = tf.math.imag(complex_spec)
spec = tf.math.sqrt(real * real + imag * imag)

The original MelSpecLayerSimple.py likely used:
spec = tf.abs(complex_spec)  # This doesn't work with CoreML!

CoreML doesn't support the ComplexAbs operation that TensorFlow
generates when you call tf.abs() on complex numbers. The "fixed"
version manually computes the magnitude using the mathematical
formula |z| = sqrt(real² + imag²), which CoreML can handle since it
  only uses basic arithmetic operations.

Evolution of BirdNET models

It appears BirdNET evolved from:
1. Early versions: Used SimpleSpecLayer (basic spectrograms)
2. Later versions: Switched to MelSpecLayerSimple (mel
spectrograms) for better performance
3. CoreML conversion: Required the "fixed" version to avoid
unsupported operations

This evolution makes sense because mel spectrograms are generally
superior for audio ML tasks - they better match how humans (and
likely birds) perceive frequency differences, with higher
resolution in lower frequencies where most biological sounds occur.

## Acknowledgments

- [BirdNET](https://github.com/kahst/BirdNET-Analyzer) team for the original models
- [BirdNET-CoreML](https://github.com/kahst/BirdNET-CoreML) (No longer public, under MIT)

# BirdNET-CoreML

Convert trained BirdNET Keras models into Apple Core ML packages for on-device inference.

---

## Overview

1. **Build** a BirdNET spectrogram+ResNet model in Keras.  
2. **Export** it as a single-signature TensorFlow SavedModel.  
3. **Convert** that SavedModel into an Apple Core ML Program package (`.mlpackage`).

---

## Requirements

- **macOS** (Core ML uses Apple’s frameworks)  
- **Python ≥ 3.10**  
- **Homebrew** (optional, for installing Python 3.10/3.11)  
- **Git**  

---

## Getting the BirdNET checkpoints

This converter expects to find the raw Keras checkpoint and its JSON config
in `model/` under the CoreML project. You can grab them from the
[BirdNET-Analyzer](https://github.com/kahst/BirdNET-Analyzer) repo:

```bash
# From your project root:
git clone https://github.com/kahst/BirdNET-Analyzer.git

# Copy the 6K-class RAW files into this repo's model/ folder
mkdir -p BirdNET-CoreML/model
cp BirdNET-Analyzer/checkpoints/BirdNET_6000_RAW_model.h5       \
   BirdNET-CoreML/model/BirdNET_6000_RAW_model.keras
cp BirdNET-Analyzer/checkpoints/BirdNET_6000_RAW_config.json    \
   BirdNET-CoreML/model/BirdNET_6000_RAW_model_config.json
```

## Setup

### Stable (coremltools 6.x on Python 3.10)

```bash
# 1. Create & activate a 3.10 virtual env
python3.10 -m venv birdnet310
source birdnet310/bin/activate

# 2. Install TensorFlow (includes Keras-2)
pip install --upgrade pip
pip install tensorflow-macos==2.13.1 tensorflow-metal==1.0.1

# 3. Install Core ML Tools 6.x
pip install coremltools<7

# 4. Freeze deps for reproducibility
pip freeze > requirements.txt
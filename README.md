# BirdNET-CoreML

Convert a trained BirdNET Keras model into an Apple Core ML Program package (`.mlpackage`) for on-device inference on iOS 15+.

**✅ VERIFIED WORKING** - The export script has been fixed and tested with TensorFlow 2.15.0 and CoreML Tools 8.3.0.

---

## Overview

1. **Build** a BirdNET spectrogram + ResNet model in Keras.  
2. **Export** it as a single-signature TensorFlow SavedModel.  
3. **Convert** that SavedModel into an Apple Core ML Program package (`.mlpackage`).

---

## Requirements

- **macOS** (Core ML uses Apple frameworks)  
- **Python 3.11+**  
- **Git**  

---

## Quick Setup (Recommended)

Use the automated setup script to create a clean environment:

```bash
cd BirdNET-CoreML
./setup_environment.sh
```

This will:
- Create a clean virtual environment (`birdnet_clean_venv`)
- Install TensorFlow 2.15.0 for macOS
- Install CoreML Tools 8.3.0
- Install all required dependencies

After setup, activate the environment:
```bash
source birdnet_clean_venv/bin/activate
```

## Manual Setup (Alternative)

1. **Clone this repo** (and the BirdNET checkpoints):

   ```bash
   git clone https://github.com/kahst/BirdNET-Analyzer.git
   git clone https://github.com/gioneill/BirdNET-CoreML.git
   ```

2. **Create & activate your Python virtual environment**:

   ```bash
   cd BirdNET-CoreML
   python3 -m venv birdnet_clean_venv
   source birdnet_clean_venv/bin/activate
   ```

3. **Install core dependencies**:

   ```bash
   pip install --upgrade pip
   pip install tensorflow-macos==2.15.0
   pip install coremltools==8.3.0
   pip install -r requirements.txt  # additional dependencies
   ```

4. **Verify your installation**:

   ```bash
   python coreml_export/export_coreml_gpt03.py --help
   ```

   You should see the help output without any import errors.

---

## Converting Models to Core ML

### Using the Fixed Export Script

The main export script `coreml_export/export_coreml_gpt03.py` supports multiple input formats:

**Convert a Keras model (.keras or .h5):**
```bash
python coreml_export/export_coreml_gpt03.py \
  --in_path path/to/model.keras \
  --out_path output/model.mlpackage
```

**Convert a TensorFlow SavedModel directory:**
```bash
python coreml_export/export_coreml_gpt03.py \
  --in_path path/to/savedmodel/ \
  --out_path output/model.mlpackage
```

**Options:**
- `--target ios15` - Minimum deployment target (ios15, macos12, tvos16)
- `--keep_fp32` - Keep weights in FP32 (default is FP16 for smaller size)

### Legacy Scripts

The original conversion scripts are still available:

**Build Keras model from checkpoint:**
```bash
python build_model.py
# → writes model/BirdNET_6000_RAW_model.keras
```

**Convert to Core ML:**
```bash
python convert.py
# → produces model/BirdNET_6000_RAW.mlpackage/
```

---

## Troubleshooting

### Import Errors
If you encounter import errors, the most likely cause is a corrupted virtual environment. Solution:
1. Delete your existing virtual environment
2. Run `./setup_environment.sh` to create a fresh one

### TensorFlow Version Warning
You may see: `TensorFlow version 2.15.0 has not been tested with coremltools. You may run into unexpected errors.`

This warning can be safely ignored. The conversion has been tested and works correctly.

### Custom Layers
The export script includes support for BirdNET's custom layers:
- `SimpleSpecLayer` - For spectrogram computation
- `MelSpecLayerSimple` - For mel-scale preprocessing

---

## Project Layout

```
project-root/
├─ BirdNET-Analyzer/                # upstream repo with checkpoints
│  └─ checkpoints/
│     ├─ BirdNET_6000_RAW_model.h5
│     └─ BirdNET_6000_RAW_config.json
└─ BirdNET-CoreML/                  # this repo
   ├─ model/
   │  ├─ BirdNET_6000_RAW_model.keras
   │  └─ BirdNET_6000_RAW_model_config.json
   ├─ coreml_export/
   │  └─ export_coreml_gpt03.py     # ✅ Fixed export script
   ├─ custom_layers.py
   ├─ build_model.py
   ├─ convert.py
   ├─ setup_environment.sh          # 🆕 Automated setup
   ├─ requirements.txt
   └─ README.md
```

---

## Technical Details

- **TensorFlow**: 2.15.0 (macOS optimized)
- **CoreML Tools**: 8.3.0
- **Input Format**: 3-second mono audio @ 48 kHz (144,000 samples)
- **Output Format**: Core ML .mlpackage for iOS 15+
- **Model Size**: ~50MB (FP16) or ~100MB (FP32)

The conversion process handles BirdNET's custom preprocessing layers and ensures compatibility with iOS Core ML runtime.

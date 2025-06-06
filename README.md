# BirdNET-CoreML

Convert a trained BirdNET Keras model into an Apple Core ML Program package (`.mlpackage`) for on-device inference on iOS 15+.

Note: This was forked from a repo that is no longer available (https://github.com/kahst/BirdNET-CoreML), and additionally, has not yet been tested to actually work. I will update this when I have a chance to test it...

---

## Overview

1. **Build** a BirdNET spectrogram + ResNet model in Keras.  
2. **Export** it as a single-signature TensorFlow SavedModel.  
3. **Convert** that SavedModel into an Apple Core ML Program package (`.mlpackage`).

---

## Requirements

- **macOS** (Core ML uses Apple frameworks)  
- **Python 3.11**  
- **Git**  

---

## Setup

1. **Clone this repo** (and the BirdNET checkpoints):

   ```bash
   git clone https://github.com/kahst/BirdNET-Analyzer.git
   git clone https://github.com/gioneill/BirdNET-CoreML.git
   ```

2. **Create & activate your Python 3.11 virtual environment**:

   ```bash
   python3.11 -m venv birdnet311
   source birdnet311/bin/activate # macOS/Linux
   ```

3. **Install dependencies**:

   ```bash
   # If requirements.txt already exists:
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   ```bash
   # If you just installed packages manually and haven’t generated requirements.txt yet:
   pip freeze > requirements.txt
   ```

4. **Verify your stack**:

   ```bash
   python - <<'PY'
   import tensorflow as tf, coremltools as ct
   print("TensorFlow:", tf.__version__)
   print("coremltools:", ct.__version__)
   PY
   ```
   should return:
   ```
   TensorFlow: 2.19.0
   coremltools: 8.3.0
   ```

---

## Building the Keras Model

Edit hyperparameters in build_model.py if needed (e.g. NUM_CLASSES = 6522), then run:

```bash
cd BirdNET-CoreML
python build_model.py
# → writes model/BirdNET_6000_RAW_model.keras
```

---

## Converting to Core ML

For iOS 15+ (ML Program format):

```bash
cd BirdNET-CoreML
python convert.py
# → produces model/BirdNET_6000_RAW.mlpackage/
```

---

Project Layout

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
   ├─ custom_layers.py
   ├─ build_model.py
   ├─ convert.py
   ├─ requirements.txt
   └─ README.md
```

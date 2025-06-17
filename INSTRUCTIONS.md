# BirdNET-CoreML Setup Instructions

## Quick Start

1. **Automated Setup (Recommended)**:
   ```bash
   cd BirdNET-CoreML
   ./setup_environment.sh
   source birdnet_clean_venv/bin/activate
   ```

2. **Test the Fixed Export Script**:
   ```bash
   python coreml_export/export_coreml_gpt03.py --help
   ```

3. **Convert a Model**:
   ```bash
   python coreml_export/export_coreml_gpt03.py \
     --in_path path/to/model.keras \
     --out_path output/model.mlpackage
   ```

## What Was Fixed

- ✅ Fixed Keras 3 vs Keras 2.15 compatibility issues
- ✅ Created clean virtual environment with working dependencies
- ✅ Updated model loading to work with TensorFlow 2.15.0
- ✅ Added automated setup script
- ✅ Updated documentation with working instructions

## Dependencies (Verified Working)

- Python 3.11+
- TensorFlow 2.15.0 (macOS)
- CoreML Tools 8.3.0
- Keras 2.15.0 (included with TensorFlow)

## Environment

Use the `birdnet_clean_venv` environment created by the setup script. The old environments (`birdnet_new_venv`, `birdval`) had corrupted installations and should be avoided.

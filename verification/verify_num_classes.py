#!/usr/bin/env python3
"""
verify_num_classes.py

This script checks that the number of output classes in the BirdNET CoreML model
used by test_birdnet_simple.py is exactly 6522.

It does so by:
  1. Loading the CoreML model via coremltools.
  2. Performing a single dummy forward pass (zeros) to extract "classLabel_probs".
  3. Asserting that the number of keys in classLabel_probs is 6522.

Usage:
  python3 verify_num_classes.py
"""

import sys
import numpy as np

try:
    import coremltools as ct
except ImportError:
    print("❌ coremltools is required. Install with 'pip install coremltools'")
    sys.exit(1)

MODEL_PATH = "/Users/grego/Developer/BirdNET/coreml_models/BirdNET_6000_RAW_WITHLABELS.mlpackage/Data/com.apple.CoreML/model.mlmodel"
EXPECTED_NUM_CLASSES = 6522

def verify_num_classes(model_path: str, expected: int) -> None:
    print(f"Loading Core ML model from: {model_path}")
    try:
        mlmodel = ct.models.MLModel(model_path)
    except Exception as e:
        print(f"❌ Failed to load Core ML model: {e}")
        sys.exit(1)

    dummy_input = np.zeros((1, 144000), dtype=np.float32)
    try:
        print("Running a dummy inference to extract 'classLabel_probs' …")
        output = mlmodel.predict({"input": dummy_input})
    except Exception as e:
        print(f"❌ Failed to run dummy predict: {e}")
        sys.exit(1)

    if "classLabel_probs" not in output:
        print("❌ The model's output dictionary does not contain 'classLabel_probs'.")
        print("   Available keys:", list(output.keys()))
        sys.exit(1)

    probs_dict = output["classLabel_probs"]
    if not isinstance(probs_dict, dict):
        print("❌ Expected 'classLabel_probs' to be a dict(String→Double), but got:", type(probs_dict))
        sys.exit(1)

    num_classes = len(probs_dict)
    print(f"Model returned {num_classes} classes in 'classLabel_probs'")

    if num_classes == expected:
        print(f"✅ PASS: Model has the expected number of classes ({expected}).")
        sys.exit(0)
    else:
        print(f"❌ FAIL: Model has {num_classes} classes, expected {expected}.")
        sys.exit(1)

if __name__ == "__main__":
    verify_num_classes(MODEL_PATH, EXPECTED_NUM_CLASSES)

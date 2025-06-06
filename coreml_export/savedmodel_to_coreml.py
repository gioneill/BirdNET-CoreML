#!/usr/bin/env python3
"""
Convert a TensorFlow SavedModel or Keras .h5 model to CoreML format using coremltools.

Usage:
    python savedmodel_to_coreml.py --savedmodel_dir PATH_TO_SAVEDMODEL_OR_H5 --labels PATH_TO_LABELS --output PATH_TO_MLMODEL --custom_layer PATH_TO_PY

This script loads a TensorFlow SavedModel or Keras .h5, converts it to CoreML, and saves the .mlmodel file.
"""

import argparse
import coremltools as ct
from pathlib import Path
import importlib.util
import sys
import tensorflow as tf

def import_custom_layer(py_path, class_name):
    spec = importlib.util.spec_from_file_location(class_name, py_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[class_name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)

def main():
    parser = argparse.ArgumentParser(description="Convert TensorFlow SavedModel or Keras .h5 to CoreML .mlmodel")
    parser.add_argument("--savedmodel_dir", type=str, required=True, help="Path to the TensorFlow SavedModel directory or Keras .h5 file")
    parser.add_argument("--labels", type=str, required=True, help="Path to class labels .txt file (one label per line)")
    parser.add_argument("--output", type=str, required=True, help="Path to output .mlmodel file")
    parser.add_argument("--custom_layer", type=str, required=False, help="Path to custom layer .py file (if needed)")
    parser.add_argument("--ios_target", type=str, default="iOS15", help="Minimum deployment target (e.g., iOS15, macOS14)")
    args = parser.parse_args()

    # Load class labels
    with open(args.labels, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]

    classifier_cfg = ct.ClassifierConfig(class_labels=labels)

    # Load the model with custom layer if needed
    if args.custom_layer:
        MelSpecLayerSimple = import_custom_layer(args.custom_layer, "MelSpecLayerSimple")
        model = tf.keras.models.load_model(args.savedmodel_dir, custom_objects={"MelSpecLayerSimple": MelSpecLayerSimple})
    else:
        model = args.savedmodel_dir

    print(f"Converting model at {args.savedmodel_dir} to CoreML...")
    mlmodel = ct.convert(
        model,
        source="tensorflow",
        classifier_config=classifier_cfg,
        minimum_deployment_target=getattr(ct.target, args.ios_target),
        compute_units=ct.ComputeUnit.ALL,
    )
    mlmodel.author = "BirdNET Team (converted via coremltools)"
    mlmodel.short_description = "Bird sound recognition with BirdNET (CoreML export pipeline)."
    mlmodel.save(args.output)
    print(f"CoreML model saved to: {args.output}")

if __name__ == "__main__":
    main()

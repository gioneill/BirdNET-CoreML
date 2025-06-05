#!/usr/bin/env python3
"""
Convert a TensorFlow SavedModel to CoreML format using coremltools.

Usage:
    python savedmodel_to_coreml.py --savedmodel_dir PATH_TO_SAVEDMODEL --labels PATH_TO_LABELS --output PATH_TO_MLMODEL

This script loads a TensorFlow SavedModel, converts it to CoreML, and saves the .mlmodel file.
"""

import argparse
import coremltools as ct
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Convert TensorFlow SavedModel to CoreML .mlmodel")
    parser.add_argument("--savedmodel_dir", type=str, required=True, help="Path to the TensorFlow SavedModel directory")
    parser.add_argument("--labels", type=str, required=True, help="Path to class labels .txt file (one label per line)")
    parser.add_argument("--output", type=str, required=True, help="Path to output .mlmodel file")
    parser.add_argument("--ios_target", type=str, default="iOS15", help="Minimum deployment target (e.g., iOS15, macOS14)")
    args = parser.parse_args()

    # Load class labels
    with open(args.labels, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]

    classifier_cfg = ct.ClassifierConfig(class_labels=labels)

    print(f"Converting SavedModel at {args.savedmodel_dir} to CoreML...")
    mlmodel = ct.convert(
        args.savedmodel_dir,
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

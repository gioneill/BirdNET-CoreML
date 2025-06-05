#!/usr/bin/env python3
"""
Strip a TensorFlow SavedModel to a single signature for CoreML conversion.

Usage:
    python strip_savedmodel_to_single_signature.py --input_dir PATH_TO_ORIG_SAVEDMODEL --output_dir PATH_TO_SINGLE_SIG_SAVEDMODEL

This script loads a SavedModel, extracts the default serving signature, and saves a new SavedModel with only that signature.
"""

import argparse
import tensorflow as tf
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Strip a SavedModel to a single signature for CoreML conversion")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to original SavedModel directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output single-signature SavedModel directory")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model = tf.saved_model.load(input_dir)
    # Get the default serving signature
    if "serving_default" not in model.signatures:
        raise RuntimeError("No 'serving_default' signature found in the SavedModel.")
    serving_fn = model.signatures["serving_default"]

    # Save a new SavedModel with only the serving_default signature
    tf.saved_model.save(
        model,
        output_dir,
        signatures={"serving_default": serving_fn}
    )
    print(f"Single-signature SavedModel exported to: {output_dir}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Add a 'serving_default' signature to a TensorFlow SavedModel for CoreML conversion.

Usage:
    python add_signature_to_savedmodel.py --input_dir PATH_TO_ORIG_SAVEDMODEL --output_dir PATH_TO_SIGNED_SAVEDMODEL

This script loads a SavedModel using tf.saved_model.load, wraps the callable in a tf.function with an explicit input signature, and saves a new SavedModel with the 'serving_default' signature.
"""

import argparse
import tensorflow as tf
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Add a 'serving_default' signature to a SavedModel")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to original SavedModel directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output signed SavedModel directory")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model = tf.saved_model.load(input_dir)
    # Try to infer input shape and dtype from the model's callable
    # We'll use the first function in the model's __call__ signatures
    concrete_fns = list(model.signatures.values())
    if concrete_fns:
        # Already has a signature, just re-save with only this one
        serving_fn = concrete_fns[0]
        tf.saved_model.save(model, output_dir, signatures={"serving_default": serving_fn})
        print(f"SavedModel with 'serving_default' signature exported to: {output_dir}")
        return

    # If no signatures, try to infer from the callable
    # This is a fallback and may need to be adjusted for your model
    # Here we assume input shape (None, 144000) and dtype float32
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 144000], dtype=tf.float32, name="input")])
    def serving_fn(x):
        return {"output": model(x)}

    tf.saved_model.save(
        model,
        output_dir,
        signatures={"serving_default": serving_fn.get_concrete_function()}
    )
    print(f"SavedModel with 'serving_default' signature exported to: {output_dir}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Load a Keras model with a custom layer and save it as a new .h5 or SavedModel for CoreML conversion.

Usage:
    python prepare_keras_for_coreml.py --input_model PATH_TO_H5 --custom_layer PATH_TO_PY --output_model PATH_TO_OUTPUT

This script loads a Keras model with a custom layer (MelSpecLayerSimple), then saves it as a new .h5 file.
"""

import argparse
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
    import os
    
    # Default path to the fixed MelSpecLayerSimple
    default_layer_path = os.path.join(os.path.dirname(__file__), "input", "MelSpecLayerSimple_fixed.py")
    
    parser = argparse.ArgumentParser(description="Prepare Keras model with custom layer for CoreML conversion")
    parser.add_argument("--input_model", type=str, required=True, help="Path to input Keras .h5 model")
    parser.add_argument("--custom_layer", type=str, default=default_layer_path, help=f"Path to custom layer .py file (default: {default_layer_path})")
    parser.add_argument("--output_model", type=str, required=True, help="Path to output .h5 model")
    args = parser.parse_args()

    # Import the custom layer
    MelSpecLayerSimple = import_custom_layer(args.custom_layer, "MelSpecLayerSimple")

    # Load the model with the custom layer
    model = tf.keras.models.load_model(args.input_model, custom_objects={"MelSpecLayerSimple": MelSpecLayerSimple})

    # Save as a new .h5 file (could also save as SavedModel if needed)
    model.save(args.output_model)
    print(f"Model with custom layer saved to: {args.output_model}")

if __name__ == "__main__":
    main()

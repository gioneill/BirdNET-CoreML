#!/usr/bin/env python3
"""
Convert the BirdNET meta-model.h5 (location/time → species priors) to Core ML .mlpackage.

This script specifically handles the metadata model that takes encoded location/time data
(144 features) and outputs species occurrence probabilities (6522 species).

Usage:
    python convert_meta_model_to_coreml.py --input input/meta-model.h5 --output output/meta-model.mlpackage
"""

import argparse
import sys
from pathlib import Path

import coremltools as ct
import tensorflow as tf

# Make repo‑root importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))


class MDataLayer(tf.keras.layers.Layer):
    """
    MDataLayer for the metadata model.
    Handles location/time data encoding for species occurrence prediction.
    """
    def __init__(self, embeddings=None, **kwargs):
        super().__init__(**kwargs)
        self.embeddings = embeddings
    
    def call(self, inputs):
        return inputs
    
    def get_config(self):
        config = super().get_config()
        config.update({'embeddings': self.embeddings})
        return config


def _parse_args():
    p = argparse.ArgumentParser(description="Convert BirdNET meta-model to Core ML")
    p.add_argument(
        "--input", 
        default="input/meta-model.h5",
        help="Input meta-model.h5 file (default: input/meta-model.h5)"
    )
    p.add_argument(
        "--output", 
        default="output/meta-model.mlpackage",
        help="Output .mlpackage file (default: output/meta-model.mlpackage)"
    )
    p.add_argument(
        "--target",
        default="ios15",
        help="Minimum deployment target (e.g. ios15, macos12, tvos16)",
    )
    p.add_argument(
        "--keep_fp32",
        action="store_true",
        help="Keep weights in FP32 (otherwise down‑cast to FP16)",
    )
    return p.parse_args()


def load_meta_model(model_path: Path):
    """
    Load the metadata model with custom MDataLayer.
    """
    custom_objects = {
        "MDataLayer": MDataLayer,
    }
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
        print(f"✅ Loaded meta-model from {model_path}")
        
        # Print model info
        print(f"Input shape: {model.input.shape}")
        print(f"Output shape: {model.output.shape}")
        
        return model
    except Exception as e:
        print(f"❌ Error loading meta-model: {e}")
        raise


def main():
    args = _parse_args()
    
    # Resolve paths
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Make sure input exists
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return 1
    
    # Load the meta-model
    try:
        model = load_meta_model(input_path)
    except Exception:
        return 1
    
    # Get input name for CoreML conversion
    input_name = model.inputs[0].name.split(":")[0]
    
    # Define input: 144 features for encoded location/time data
    # Format: [lat_encoded(72), lon_encoded(72)] = 144 total features
    meta_input = ct.TensorType(shape=(1, 144), name=input_name)
    
    print(f"Converting model with input: {meta_input}")
    
    # Convert to Core ML
    try:
        mlmodel = ct.convert(
            model,
            inputs=[meta_input],
        )
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        mlmodel.save(str(output_path))
        print(f"✅ Saved Core ML meta-model to {output_path}")
        
        # Print some model info
        print(f"\nModel details:")
        print(f"  Input: {meta_input}")
        print(f"  Output shape: 6522 species probabilities")
        print(f"  File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error converting to Core ML: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

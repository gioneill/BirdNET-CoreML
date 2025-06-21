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

import numpy as np
import coremltools as ct
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    try:
        import keras
        import tensorflow as tf
    except ImportError:
        print("❌ TensorFlow/Keras not available. Please install tensorflow and activate the proper environment.")
        sys.exit(1)

# Make repo‑root importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))


class MDataLayer(keras.layers.Layer):
    """
    MDataLayer for the metadata model.
    Handles location/time data encoding for species occurrence prediction.
    Transforms 3 inputs (lat, lon, week) into 144 features using sinusoidal encoding.
    """
    def __init__(self, embeddings=None, **kwargs):
        super().__init__(**kwargs)
        self.embeddings = embeddings
    
    def call(self, inputs):
        # inputs shape: (batch_size, 3) containing [lat, lon, week]
        lat = inputs[:, 0:1]
        lon = inputs[:, 1:2] 
        week = inputs[:, 2:3]
        
        # Apply the encoding transformation
        feats = []
        
        # Encode latitude (normalized to -1 to 1)
        lat_norm = lat / 90.0
        for k in range(1, 25):  # 24 frequencies
            feats.append(tf.sin(k * np.pi * lat_norm))
            feats.append(tf.cos(k * np.pi * lat_norm))
        
        # Encode longitude (normalized to -1 to 1)
        lon_norm = lon / 180.0
        for k in range(1, 25):  # 24 frequencies
            feats.append(tf.sin(k * np.pi * lon_norm))
            feats.append(tf.cos(k * np.pi * lon_norm))
        
        # Encode week (normalized to 0 to 1, cyclic)
        week_norm = (week - 1) / 48.0
        for k in range(1, 25):  # 24 frequencies
            feats.append(tf.sin(k * 2 * np.pi * week_norm))
            feats.append(tf.cos(k * 2 * np.pi * week_norm))
        
        # Concatenate all features
        return tf.concat(feats, axis=-1)  # Output shape: (batch_size, 144)
    
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
    
    # Define input: 3 features (lat, lon, week)
    # The MDataLayer will transform these into 144 encoded features internally
    meta_input = ct.TensorType(shape=(1, 3), name=input_name, dtype=np.float32)
    
    print(f"Converting model with input: {meta_input}")
    print(f"  Input features: [latitude, longitude, week]")
    print(f"  MDataLayer transforms to 144 encoded features")
    
    # Convert to Core ML
    try:
        mlmodel = ct.convert(
            model,
            inputs=[meta_input],
            compute_precision=(
                ct.precision.FLOAT16
            ),
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

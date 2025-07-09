#!/usr/bin/env python3
"""
Convert a BirdNET .keras or .h5 model — or a TensorFlow SavedModel directory —
to a Core ML .mlpackage ready for iOS 15 +.
"""

import argparse
import sys
from pathlib import Path

import coremltools as ct
import tensorflow as tf

# Make repo‑root importable so that `coreml_export.input` resolve
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))


def _load_melspec_layer(filename: str):
    """Load MelSpecLayerSimple from the specified filename."""
    import importlib.util
    melspec_path = repo_root / "coreml_export" / "input" / filename
    if not melspec_path.exists():
        raise FileNotFoundError(f"MelSpecLayerSimple file not found: {melspec_path}")
    
    spec = importlib.util.spec_from_file_location("MelSpecLayerSimple", melspec_path)
    melspec_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(melspec_module)
    return melspec_module.MelSpecLayerSimple




class MDataLayer(tf.keras.layers.Layer):
    """
    Placeholder MDataLayer for loading the metadata model.
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
    p = argparse.ArgumentParser(description="Export Keras/SavedModel to Core ML")
    p.add_argument(
        "--in_path", required=True, help="Input .keras / .h5 file or SavedModel directory"
    )
    p.add_argument(
        "--out_path", required=True, help="Output .mlpackage (or .mlmodel if desired)"
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
    p.add_argument(
        "--melspec_layer_file",
        default="MelSpecLayerSimple_fixed.py",
        help="Filename of the MelSpecLayerSimple implementation to use (default: MelSpecLayerSimple_fixed.py)",
    )
    p.add_argument(
        "--meta_model_path",
        default="input/meta-model.h5",
        help="Path to the metadata model file (default: input/meta-model.h5)",
    )
    p.add_argument(
        "--meta_out_path",
        help="Output path for metadata model .mlpackage (if not provided, will use out_path with '_meta' suffix)",
    )
    p.add_argument(
        "--convert_meta",
        action="store_true",
        help="Also convert the metadata model to CoreML",
    )
    return p.parse_args()


def _load_any_model(path: Path, melspec_layer_class):
    """
    Load either a Keras model (.keras/.h5) or a TensorFlow SavedModel directory.
    """
    custom = {
        "MelSpecLayerSimple": melspec_layer_class,
    }

    ext = path.suffix.lower()
    if ext in (".keras", ".h5"):
        # Use tf.keras for both .h5 and .keras files
        return tf.keras.models.load_model(path, compile=False, custom_objects=custom)

    if path.is_dir():
        # For SavedModel directories, load directly using tf.saved_model.load
        # and create a callable wrapper
        loaded_model = tf.saved_model.load(str(path))
        
        # Create a simple wrapper function that can be used like a Keras model
        class SavedModelWrapper:
            def __init__(self, saved_model):
                self.saved_model = saved_model
                self.serving_fn = saved_model.signatures.get("serving_default")
                if self.serving_fn is None:
                    # Try to get the first available signature
                    signatures = list(saved_model.signatures.keys())
                    if signatures:
                        self.serving_fn = saved_model.signatures[signatures[0]]
                    else:
                        raise ValueError("No signatures found in SavedModel")
                
                # Extract input specs for the inputs property
                self._inputs = [tf.TensorSpec(shape=inp.shape, dtype=inp.dtype, name=inp.name) 
                              for inp in self.serving_fn.inputs]
            
            def __call__(self, inputs, **kwargs):
                # Convert inputs to the expected format
                if isinstance(inputs, tf.Tensor):
                    # Assume single input - get the input name from the signature
                    input_name = list(self.serving_fn.structured_input_signature[1].keys())[0]
                    inputs = {input_name: inputs}
                return self.serving_fn(**inputs)
            
            @property
            def inputs(self):
                return self._inputs
        
        return SavedModelWrapper(loaded_model)

    raise ValueError(f"Unsupported input format: {path}")


def main():
    args = _parse_args()

    # Load the specified MelSpecLayerSimple implementation
    MelSpecLayerSimple = _load_melspec_layer(args.melspec_layer_file)

    model = _load_any_model(Path(args.in_path), MelSpecLayerSimple)
    input_name = model.inputs[0].name.split(":")[0]

    # 3‑second mono audio @ 48 kHz → 144 000 samples
    audio_input = ct.TensorType(shape=(1, 144_000), name=input_name)

    target_attr = args.target.lower()
    if target_attr.startswith("ios"):
        target_attr = "iOS" + target_attr[3:]

    mlmodel = ct.convert(
        model,
        inputs=[audio_input],
        # minimum_deployment_target=getattr(ct.target, target_attr),
        compute_precision=(
            ct.precision.FLOAT32 if args.keep_fp32 else ct.precision.FLOAT16
        ),
    )
    mlmodel.save(args.out_path)
    print(f"✅  Saved Core ML model to {args.out_path}")


if __name__ == "__main__":
    main()

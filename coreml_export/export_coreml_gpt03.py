#!/usr/bin/env python3
"""
Convert a BirdNET .keras or legacy .h5 model — or a SavedModel folder —
to a Core ML .mlpackage ready for iOS 15+.
"""
import argparse
from pathlib import Path
import coremltools as ct
import tensorflow as tf
import sys

# Ensure the repo root is importable so custom_layers and coreml_export.input can be found
sys.path.append(str(Path(__file__).resolve().parent.parent))
from custom_layers import SimpleSpecLayer
from coreml_export.input.MelSpecLayerSimple import MelSpecLayerSimple

class LegacySimpleSpecLayer(SimpleSpecLayer):
    """
    Wrapper handling legacy spectrogram layer with fmin/fmax parameters.
    """
    def __init__(self, sample_rate=48000, spec_shape=(257,384),
                 frame_step=374, frame_length=512,
                 fmin=0, fmax=3000, data_format='channels_last', **kwargs):
        super().__init__(sample_rate=sample_rate,
                         spec_shape=spec_shape,
                         frame_step=frame_step,
                         frame_length=frame_length,
                         data_format=data_format,
                         **kwargs)
        self.fmin = fmin
        self.fmax = fmax
    def get_config(self):
        cfg = super().get_config()
        cfg.update({'fmin': self.fmin, 'fmax': self.fmax})
        return cfg

def parse_args():
    p = argparse.ArgumentParser(description="Export Keras/SavedModel to CoreML")
    p.add_argument("--in_path",  required=True, help="Input .keras/.h5 file or SavedModel dir")
    p.add_argument("--out_path", required=True, help="Output .mlpackage or .mlmodel")
    p.add_argument("--target",   default="ios15", help="Min deployment target (ios15, macos12, etc.)")
    p.add_argument("--keep_fp32", action="store_true", help="Keep weights in FP32 (no FP16 downcast)")
    return p.parse_args()

def load_any_model(path: Path):
    ext = path.suffix.lower()
    custom_objects = {
        "SimpleSpecLayer": LegacySimpleSpecLayer,
        "MelSpecLayerSimple": MelSpecLayerSimple
    }
    if ext in (".keras", ".h5"):
        loader = tf.keras.models.load_model if ext == ".h5" else __import__("keras").saving.load_model
        return loader(path, compile=False, custom_objects=custom_objects)
    if path.is_dir():
        from keras.layers import TFSMLayer
        layer = TFSMLayer(path, call_endpoint="serving_default")
        return tf.keras.Sequential([layer])
    raise ValueError(f"Unsupported input format: {path}")

def main():
    args = parse_args()
    model = load_any_model(Path(args.in_path))
    input_name = model.inputs[0].name.split(":")[0]
    audio_input = ct.TensorType(shape=(1,144000), name=input_name)
    target_attr = args.target.lower()
    if target_attr.startswith("ios"):
        target_attr = "iOS" + target_attr[3:]
    mlmodel = ct.convert(
        model,
        inputs=[audio_input],
        minimum_deployment_target=getattr(ct.target, target_attr),
        compute_precision=(ct.precision.FLOAT32 if args.keep_fp32 else ct.precision.FLOAT16),
    )
    mlmodel.save(args.out_path)
    print(f"✅ Saved Core ML model to {args.out_path}")

if __name__ == "__main__":
    main()

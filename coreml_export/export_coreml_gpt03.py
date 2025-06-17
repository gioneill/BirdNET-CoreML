
#!/usr/bin/env python3
"""
Convert a BirdNET .keras or .h5 model — or a TensorFlow SavedModel directory —
to a Core ML .mlpackage ready for iOS 15 +.
"""

import argparse
import sys
from pathlib import Path

import coremltools as ct
import tensorflow as tf
import keras  # standalone Keras 3 API

# Make repo‑root importable so that `custom_layers` and `coreml_export.input` resolve
sys.path.append(str(Path(__file__).resolve().parent.parent))

from custom_layers import SimpleSpecLayer
from coreml_export.input.MelSpecLayerSimple import MelSpecLayerSimple


class LegacySimpleSpecLayer(SimpleSpecLayer):
    """
    Wrapper layer that keeps compatibility with very old checkpoints
    that passed `fmin` / `fmax` into the constructor.
    """

    def __init__(
        self,
        sample_rate: int = 48_000,
        spec_shape: tuple[int, int] = (257, 384),
        frame_step: int = 374,
        frame_length: int = 512,
        fmin: int = 0,
        fmax: int = 3_000,
        data_format: str = "channels_last",
        **kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            spec_shape=spec_shape,
            frame_step=frame_step,
            frame_length=frame_length,
            data_format=data_format,
            **kwargs,
        )
        self.fmin = fmin
        self.fmax = fmax

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"fmin": self.fmin, "fmax": self.fmax})
        return cfg


def _parse_args():
    p = argparse.ArgumentParser(description="Export Keras/SavedModel to Core ML")
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
    return p.parse_args()


def _load_any_model(path: Path):
    """
    Load either a Keras model (.keras/.h5) or a TensorFlow SavedModel directory.
    """
    custom = {
        "SimpleSpecLayer": LegacySimpleSpecLayer,
        "MelSpecLayerSimple": MelSpecLayerSimple,
    }

    ext = path.suffix.lower()
    if ext in (".keras", ".h5"):
        if ext == ".h5":
            return tf.keras.models.load_model(path, compile=False, custom_objects=custom)
        # .keras uses the Keras 3 saving API
        return keras.saving.load_model(path, compile=False, custom_objects=custom)

    if path.is_dir():
        # Wrap a SavedModel as a single TFSMLayer so we can feed it to coremltools
        from keras.layers import TFSMLayer

        layer = TFSMLayer(str(path), call_endpoint="serving_default")
        return tf.keras.Sequential([layer])

    raise ValueError(f"Unsupported input format: {path}")


def main():
    args = _parse_args()

    model = _load_any_model(Path(args.in_path))
    input_name = model.inputs[0].name.split(":")[0]

    # 3‑second mono audio @ 48 kHz → 144 000 samples
    audio_input = ct.TensorType(shape=(1, 144_000), name=input_name)

    target_attr = args.target.lower()
    if target_attr.startswith("ios"):
        target_attr = "iOS" + target_attr[3:]

    mlmodel = ct.convert(
        model,
        inputs=[audio_input],
        minimum_deployment_target=getattr(ct.target, target_attr),
        compute_precision=(
            ct.precision.FLOAT32 if args.keep_fp32 else ct.precision.FLOAT16
        ),
    )
    mlmodel.save(args.out_path)
    print(f"✅  Saved Core ML model to {args.out_path}")


if __name__ == "__main__":
    main()

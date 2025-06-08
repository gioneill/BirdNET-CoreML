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

# Ensure the repo root is importable so custom_layers can be found
sys.path.append(str(Path(__file__).resolve().parent.parent))
from custom_layers import SimpleSpecLayer  # Import after adjusting sys.path

# --- Custom Layer Classes for Model Loading -----------------------------------------------

class MelSpecLayerSimple(tf.keras.layers.Layer):
    def __init__(
        self,
        sample_rate=48000,
        spec_shape=(96, 512),
        frame_step=278,
        frame_length=2048,
        fmin=0,
        fmax=3000,
        data_format='channels_last',
        mel_filterbank=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.spec_shape = spec_shape
        self.data_format = data_format
        self.frame_step = frame_step
        self.frame_length = frame_length
        self.fmin = fmin
        self.fmax = fmax
        # Compute mel filterbank matrix
        self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.spec_shape[0],
            num_spectrogram_bins=self.frame_length // 2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.fmin,
            upper_edge_hertz=self.fmax,
            dtype=tf.float32
        )

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop('mel_filterbank', None)
        return super().from_config(config)

    def build(self, input_shape):
        # A trainable magnitude scaling parameter
        self.mag_scale = self.add_weight(
            name='magnitude_scaling',
            initializer=tf.keras.initializers.Constant(value=1.23),
            trainable=True
        )
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            return tf.TensorShape((None, self.spec_shape[0], self.spec_shape[1], 1))
        else:
            return tf.TensorShape((None, 1, self.spec_shape[0], self.spec_shape[1]))

    def call(self, inputs, training=None):
        # Normalize waveform to [-1, 1]
        inputs = inputs - tf.reduce_min(inputs, axis=1, keepdims=True)
        inputs = inputs / (tf.reduce_max(inputs, axis=1, keepdims=True) + 1e-6)
        inputs = inputs - 0.5
        inputs = inputs * 2.0

        # Perform STFT (returns complex64)
        spec = tf.signal.stft(
            inputs,
            self.frame_length,
            self.frame_step,
            fft_length=self.frame_length,
            window_fn=tf.signal.hann_window,
            pad_end=False,
            name='stft'
        )

        # Convert complex STFT output to magnitude
        real = tf.math.real(spec)
        imag = tf.math.imag(spec)
        spec = tf.sqrt(tf.square(real) + tf.square(imag))

        # After STFT:
        # For input length 144000, frame_length=2048, frame_step=278:
        #   num_frames = 1 + (144000 - 2048) // 278 = 514
        #   fft_bins = 2048 // 2 + 1 = 1025
        # So spec shape is (batch, 514, 1025)

        # Apply mel filterbank: spec = (batch, 514, num_mel_bins) with num_mel_bins = 96
        spec = tf.tensordot(spec, self.mel_filterbank, 1)

        # Convert to power spectrogram
        spec = tf.pow(spec, 2.0)

        # Nonlinear scaling
        spec = tf.pow(spec, 1.0 / (1.0 + tf.exp(self.mag_scale)))

        # Flip mel bins horizontally (axis=2)
        spec = tf.reverse(spec, axis=[2])

        # Transpose to (batch, mel_bins, frames)
        spec = tf.transpose(spec, [0, 2, 1])

        # Add channel axis
        if self.data_format == 'channels_last':
            spec = tf.expand_dims(spec, -1)  # (batch, mel_bins, frames, 1)
        else:
            spec = tf.expand_dims(spec, 1)   # (batch, 1, mel_bins, frames)

        # Final output shape is (batch, 96, 514, 1)
        return spec

    def get_config(self):
        config = {
            'data_format': self.data_format,
            'sample_rate': self.sample_rate,
            'spec_shape': self.spec_shape,
            'frame_step': self.frame_step,
            'frame_length': self.frame_length,
            'fmin': self.fmin,
            'fmax': self.fmax
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LegacySimpleSpecLayer(SimpleSpecLayer):
    """
    Wrapper that handles the legacy SimpleSpecLayer with fmin/fmax parameters.
    """
    def __init__(
        self,
        sample_rate=48000,
        spec_shape=(257, 384),
        frame_step=374,
        frame_length=512,
        fmin=0,
        fmax=3000,
        data_format='channels_last',
        **kwargs
    ):
        super().__init__(
            sample_rate=sample_rate,
            spec_shape=spec_shape,
            frame_step=frame_step,
            frame_length=frame_length,
            data_format=data_format,
            **kwargs
        )
        # Store fmin/fmax even if not used downstream
        self.fmin = fmin
        self.fmax = fmax

    def get_config(self):
        config = super().get_config()
        config.update({'fmin': self.fmin, 'fmax': self.fmax})
        return config

# --------------------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in_path",
        required=True,
        help="Model file (.keras / .h5) or SavedModel directory"
    )
    p.add_argument(
        "--out_path",
        required=True,
        help="Destination .mlpackage (or .mlmodel) path"
    )
    p.add_argument(
        "--target",
        default="ios15",
        help="Minimum deployment target (e.g., ios15, macos12)"
    )
    p.add_argument(
        "--keep_fp32",
        action="store_true",
        help="Don’t down-cast weights to FP16"
    )
    return p.parse_args()


def load_any_model(path: Path):
    """
    Unified loader that handles:
      - .keras checkpoints
      - .h5 HDF5 files
      - SavedModel folders (via TFSMLayer)
    """
    ext = path.suffix.lower()
    if ext == ".keras":
        import keras
        try:
            return keras.saving.load_model(
                path,
                compile=False,
                custom_objects={
                    "SimpleSpecLayer": LegacySimpleSpecLayer,
                    "MelSpecLayerSimple": MelSpecLayerSimple,
                }
            )
        except (AttributeError, ImportError):
            return keras.models.load_model(
                path,
                compile=False,
                custom_objects={
                    "SimpleSpecLayer": LegacySimpleSpecLayer,
                    "MelSpecLayerSimple": MelSpecLayerSimple,
                }
            )

    if ext == ".h5":
        return tf.keras.models.load_model(
            path,
            compile=False,
            custom_objects={
                "SimpleSpecLayer": LegacySimpleSpecLayer,
                "MelSpecLayerSimple": MelSpecLayerSimple,
            },
            safe_mode=False  # disable symbol whitelist
        )

    if path.is_dir():
        from keras.layers import TFSMLayer
        layer = TFSMLayer(path, call_endpoint="serving_default")
        return tf.keras.Sequential([layer])

    raise ValueError(f"Unsupported input format: {path}")


def main():
    args = parse_args()
    model_path = Path(args.in_path)
    out_path = Path(args.out_path)

    print(f"🔍 Loading model from {model_path} …")
    model = load_any_model(model_path)

    print("⚙️  Converting to Core ML …")
    # Handle target case-insensitivity (iOS vs ios)
    target_attr = args.target
    if target_attr.lower().startswith('ios'):
        target_attr = 'iOS' + target_attr[3:]

    # Match the *actual* TensorFlow/Keras input name (e.g. "INPUT" or "input_1")
    input_name = model.inputs[0].name.split(":")[0]
    audio_input = ct.TensorType(shape=(1, 144000), name=input_name)

    mlmodel = ct.convert(
        model,
        inputs=[audio_input],  # static shape prevents symbolic FFT dims
        minimum_deployment_target=getattr(ct.target, target_attr),
        compute_precision=(
            ct.precision.FLOAT32 if args.keep_fp32 else ct.precision.FLOAT16
        ),
    )

    print(f"💾 Saving Core ML bundle to {out_path} …")
    mlmodel.save(out_path)
    print("✅ Done!")


if __name__ == "__main__":
    main()

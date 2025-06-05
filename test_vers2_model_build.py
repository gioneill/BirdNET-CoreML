#!/usr/bin/env python3
"""
convert_birdnet_coreml.py

Convert BirdNET v2.4 FP32 TFLite model -> CoreML, then
pre-process an audio file and run a quick prediction.

Usage
-----
python convert_birdnet_coreml.py \
    --tflite  BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite \
    --labels  labels.txt \
    --audio   crow.wav \
    --out     BirdNET_GLOBAL_6K_V2.4.mlpackage \
    --topk    10
"""

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf
import librosa

# ----------------------------- CLI --------------------------------- #
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--tflite", required=True, type=Path, help="model/downloaded/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite")
parser.add_argument("--labels", required=True, type=Path, help="../BirdNET-Analyzer/birdnet_analyzer/example/species_list.txt")
parser.add_argument("--audio", required=True, type=Path, help="verification/crow.wav")
parser.add_argument("--out",   default="BirdNET_GLOBAL_6K_V2.4.mlpackage", type=Path,
                    help="Destination *.mlpackage")
parser.add_argument("--ios_target", default="iOS17", choices=["iOS17", "macOS14"],
                    help="Minimum deployment target")
parser.add_argument("--topk",  default=10, type=int, help="How many results to print")
args = parser.parse_args()

# --------------------- 1. Convert to Core ML ----------------------- #
print(">> Converting TFLite → Core ML … (takes ~30 s on M-series Macs)")
with args.labels.open() as f:
    label_list = [ln.strip() for ln in f if ln.strip()]
if len(label_list) != 6522:
    raise ValueError(f"Label file should have 6 522 entries, got {len(label_list)}")

classifier_cfg = ct.ClassifierConfig(class_labels=label_list)

tflite_path = Path(args.tflite).expanduser().resolve()
if not tflite_path.is_file():
    raise FileNotFoundError(f"Cannot locate TFLite model at {tflite_path}")

try:
    print("   Running Core ML conversion...")
    mlmodel = ct.convert(
        str(tflite_path),
        source="tensorflow",
        classifier_config=classifier_cfg,
        minimum_deployment_target=getattr(ct.target, args.ios_target),
        compute_units=ct.ComputeUnit.ALL,  # use Neural Engine when available
    )
    mlmodel.author  = "BirdNET Team (converted via coremltools)"
    mlmodel.license = "CC BY-NC-SA 4.0"
    mlmodel.save(str(args.out))
    print(f"✅  Core ML package saved at: {args.out}")
except Exception as e:
    print("❌  Conversion failed:", str(e))
    exit(1)

# --------------------- 2. Pre-process audio ------------------------ #
def preprocess_audio(path: Path, sr_out: int = 48_000, dur: float = 3.0) -> np.ndarray:
    """Return a (96, 511, 2) float32 array ready for the model."""
    audio, sr_in = sf.read(path, dtype="float32", always_2d=False)
    if sr_in != sr_out:
        audio = librosa.resample(audio, orig_sr=sr_in, target_sr=sr_out)
    # mono-ify
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    # center-crop or pad to exactly 3 s
    target_len = int(sr_out * dur)
    if len(audio) < target_len:
        pad = (target_len - len(audio)) // 2
        audio = np.pad(audio, (pad, target_len - len(audio) - pad))
    else:
        start = (len(audio) - target_len) // 2
        audio = audio[start:start + target_len]
    audio = np.clip(audio, -1.0, 1.0)  # normalise as per docs

    # --- compute two MEL spectrograms (docs §Technical Details) ---- #
    def mel_spec(sig, n_fft, hop, fmin, fmax):
        S = librosa.feature.melspectrogram(
            y=sig,
            sr=sr_out,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=96,
            fmin=fmin,
            fmax=fmax,
            power=2.0,      # magnitude-squared
        )
        S = np.sqrt(S, dtype=np.float32)  # Schlüter 2018 non-linear scaling
        return S

    spec_low  = mel_spec(audio, n_fft=2048, hop=278, fmin=0,   fmax=3000)
    spec_high = mel_spec(audio, n_fft=1024, hop=280, fmin=500, fmax=15000)

    if spec_low.shape[1] != 511 or spec_high.shape[1] != 511:
        raise RuntimeError(f"Got {spec_low.shape[1]} and {spec_high.shape[1]} time frames; expected 511.")

    dual = np.stack([spec_low, spec_high], axis=-1)            # (96, 511, 2)
    return dual.astype(np.float32)
 
print(">> Computing dual-channel MEL spectrogram …")
spec = preprocess_audio(args.audio)
print("   Spectrogram shape:", spec.shape)

# --------------------- 3. Run a quick prediction ------------------- #
print(">> Running sanity prediction …")
coreml = ct.models.MLModel(args.out)
# Core ML wants channels-last by default; input name is first entry in spec
inp_name = list(coreml.input_description.keys())[0]
out_dict = coreml.predict({inp_name: spec})
# out_dict is class-probability dictionary
top = sorted(out_dict.items(), key=lambda kv: kv[1], reverse=True)[: args.topk]

print(f"\nTop-{args.topk} predictions:")
for cls, prob in top:
    print(f"{prob:7.5f}  {cls}")

print("\n✅  Conversion, preprocessing, and test inference finished.")

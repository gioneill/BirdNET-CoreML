#!/usr/bin/env python3
"""
Validate CoreML model predictions against the original TFLite model.

Usage:
    python validate_coreml_vs_tflite.py --tflite_model PATH_TO_TFLITE --coreml_model PATH_TO_MLMODEL --labels PATH_TO_LABELS --audio PATH_TO_AUDIO --topk 10

This script runs inference on the same audio file using both models and compares the top-k predictions.
"""

import argparse
import numpy as np
import soundfile as sf
import librosa
import coremltools as ct

def preprocess_audio(path, sr_out=48000, dur=3.0):
    """Return a (N,) float32 array ready for the model."""
    audio, sr_in = sf.read(path, dtype="float32", always_2d=False)
    if sr_in != sr_out:
        audio = librosa.resample(audio, orig_sr=sr_in, target_sr=sr_out)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    target_len = int(sr_out * dur)
    if len(audio) < target_len:
        pad = (target_len - len(audio)) // 2
        audio = np.pad(audio, (pad, target_len - len(audio) - pad))
    else:
        start = (len(audio) - target_len) // 2
        audio = audio[start:start + target_len]
    audio = np.clip(audio, -1.0, 1.0)
    return audio.astype(np.float32)

def run_tflite(tflite_path, audio):
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Reshape input if needed
    input_shape = input_details[0]['shape']
    if input_shape[1] != audio.shape[0]:
        raise ValueError(f"TFLite model expects input of shape {input_shape}, got {audio.shape}")
    input_data = np.expand_dims(audio, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0]

def run_coreml(coreml_path, audio):
    model = ct.models.MLModel(coreml_path)
    input_name = list(model.input_description.keys())[0]
    input_data = np.expand_dims(audio, axis=0)
    out = model.predict({input_name: input_data})
    # Find the output key with class probabilities
    for k, v in out.items():
        if isinstance(v, dict):
            return v
    # fallback: return the first output
    return list(out.values())[0]

def main():
    parser = argparse.ArgumentParser(description="Validate CoreML vs TFLite model predictions")
    parser.add_argument("--tflite_model", type=str, required=True, help="Path to TFLite model")
    parser.add_argument("--coreml_model", type=str, required=True, help="Path to CoreML .mlmodel")
    parser.add_argument("--labels", type=str, required=True, help="Path to class labels .txt file")
    parser.add_argument("--audio", type=str, required=True, help="Path to test audio file (wav)")
    parser.add_argument("--topk", type=int, default=10, help="Number of top predictions to compare")
    args = parser.parse_args()

    # Load labels
    with open(args.labels, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]

    # Preprocess audio
    audio = preprocess_audio(args.audio)

    # Run TFLite
    print("Running TFLite model...")
    tflite_probs = run_tflite(args.tflite_model, audio)
    tflite_topk = np.argsort(tflite_probs)[::-1][:args.topk]
    print("\nTFLite Top-{}:".format(args.topk))
    for i in tflite_topk:
        print(f"{labels[i]:40s} {tflite_probs[i]:.5f}")

    # Run CoreML
    print("\nRunning CoreML model...")
    coreml_probs = run_coreml(args.coreml_model, audio)
    if isinstance(coreml_probs, dict):
        # Map class labels to probabilities
        coreml_probs_arr = np.array([coreml_probs.get(lbl, 0.0) for lbl in labels])
    else:
        coreml_probs_arr = np.array(coreml_probs)
    coreml_topk = np.argsort(coreml_probs_arr)[::-1][:args.topk]
    print("\nCoreML Top-{}:".format(args.topk))
    for i in coreml_topk:
        print(f"{labels[i]:40s} {coreml_probs_arr[i]:.5f}")

    # Compare overlap
    overlap = set(tflite_topk) & set(coreml_topk)
    print(f"\nTop-{args.topk} overlap: {len(overlap)} classes")
    for i in overlap:
        print(f"{labels[i]:40s} TFLite: {tflite_probs[i]:.5f}  CoreML: {coreml_probs_arr[i]:.5f}")

if __name__ == "__main__":
    main()

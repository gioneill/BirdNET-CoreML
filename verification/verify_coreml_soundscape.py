import numpy as np
import os
import sys
import scipy.io.wavfile as wav
from scipy import signal
import coremltools
from collections import Counter
import argparse

def load_audio(audio_path, target_sr=48000):
    """Load audio using scipy and return the full audio track."""
    # Read audio file
    sr, audio = wav.read(audio_path)
    
    # Convert to float32 and normalize
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128) / 128.0
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Resample if needed
    if sr != target_sr:
        audio = signal.resample(audio, int(len(audio) * target_sr / sr))
        sr = target_sr
    
    return audio, sr

def load_labels(label_path):
    """Load labels from file."""
    labels = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # Format is "Genus_species_Name" or sometimes with additional underscores
                parts = line.split('_')
                scientific_name = parts[0] + ' ' + parts[1]  # Genus species
                common_name = '_'.join(parts[2:])  # Everything after the scientific name
                labels.append((scientific_name, common_name))
    return labels

def main():
    parser = argparse.ArgumentParser(description="Analyze a soundscape file with a CoreML model.")
    parser.add_argument('--threshold', type=float, default=0.005, help='Probability threshold for displaying species.')
    args = parser.parse_args()

    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "coreml_export/output/audio-model.mlpackage")
    audio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "soundscape.wav")
    label_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "coreml_export/input/labels/en_us.txt")
    
    print(f"Loading CoreML model from {model_path}")
    model = coremltools.models.MLModel(model_path)
    print("Model loaded.")

    # Load labels
    print(f"Loading labels from {label_path}")
    labels = load_labels(label_path)
    print(f"Loaded {len(labels)} labels")

    audio, sr = load_audio(audio_path)
    print(f"Audio loaded: {len(audio)/sr:.2f} seconds, {sr} Hz")

    segment_duration = 3.0
    segment_length = int(segment_duration * sr)
    
    input_name = model.get_spec().description.input[0].name
    output_name = model.get_spec().description.output[0].name

    # Process audio in chunks
    for i in range(0, len(audio), segment_length):
        start_time = i / sr
        end_time = (i + segment_length) / sr
        print(f"\n--- Analyzing segment: {start_time:.2f}s - {end_time:.2f}s (Threshold: {args.threshold}) ---")

        chunk = audio[i:i + segment_length]
        
        if len(chunk) < segment_length:
            chunk = np.pad(chunk, (0, segment_length - len(chunk)), 'constant')
        
        input_data = np.expand_dims(chunk, axis=0).astype(np.float32)
        
        predictions = model.predict({input_name: input_data})
        preds = predictions[output_name]

        if preds.ndim == 2 and preds.shape[0] == 1:
            preds = preds[0]

        # Get indices and values of predictions above the threshold
        top_indices = np.where(preds >= args.threshold)[0]
        
        # Sort the predictions by value (descending)
        sorted_indices = sorted(top_indices, key=lambda i: preds[i], reverse=True)
        
        if not sorted_indices:
            print("No species found above the threshold for this segment.")
        else:
            for rank, idx in enumerate(sorted_indices):
                if idx < len(labels):
                    scientific_name, common_name = labels[idx]
                    print(f"{rank+1}. {scientific_name} ({common_name}): {preds[idx]:.4f}")
                else:
                    print(f"{rank+1}. Unknown Species (Index {idx}): {preds[idx]:.4f}")

if __name__ == "__main__":
    main()

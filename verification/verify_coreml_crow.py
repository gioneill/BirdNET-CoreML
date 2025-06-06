import numpy as np
import os
import sys
import scipy.io.wavfile as wav
from scipy import signal
import coremltools

def load_audio(audio_path, target_sr=48000, segment_duration=3.0):
    """Load audio using scipy instead of librosa to avoid Numba dependency."""
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
    
    # Trim or pad to segment_duration
    segment_length = int(segment_duration * sr)
    if len(audio) < segment_length:
        audio = np.pad(audio, (0, segment_length - len(audio)), 'constant')
    else:
        audio = audio[:segment_length]
    
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
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "coreml_export/output/audio-model.mlpackage")
    audio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crow.wav")
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
    print(f"Audio stats: min={np.min(audio):.6f}, max={np.max(audio):.6f}, mean={np.mean(audio):.6f}, std={np.std(audio):.6f}")

    # Model expects shape [1, 144000]
    input_data = np.expand_dims(audio, axis=0).astype(np.float32)
    print(f"Input shape: {input_data.shape}")

    # Get input description
    input_description = model.get_spec().description.input[0]
    input_name = input_description.name

    # Make prediction
    predictions = model.predict({input_name: input_data})

    # Get output description
    output_description = model.get_spec().description.output[0]
    output_name = output_description.name
    preds = predictions[output_name]

    print(f"Raw output shape: {preds.shape}")

    # If output is 2D, squeeze batch dim
    if preds.ndim == 2 and preds.shape[0] == 1:
        preds = preds[0]

    # Print stats
    print(f"Output stats: min={np.min(preds):.6f}, max={np.max(preds):.6f}, mean={np.mean(preds):.6f}, std={np.std(preds):.6f}")

    # Top-10 indices and values
    top_indices = np.argsort(preds)[::-1][:10]
    print("Top 10 output indices and values:")
    for i, idx in enumerate(top_indices):
        if idx < len(labels):
            scientific_name, common_name = labels[idx]
            print(f"{i+1}. Index: {idx}, Value: {preds[idx]:.6f}, Species: {scientific_name} ({common_name})")
        else:
            print(f"{i+1}. Index: {idx}, Value: {preds[idx]:.6f}, Species: Unknown (index out of range)")

    # Check for crow species (Corvus)
    print("\nChecking for crow species (Corvus):")
    crow_indices = []
    for i, (scientific_name, _) in enumerate(labels):
        if scientific_name.startswith("Corvus"):
            crow_indices.append(i)
    
    if crow_indices:
        print(f"Found {len(crow_indices)} crow species in the label list")
        
        # Get predictions for crow species
        crow_preds = [(i, preds[i], labels[i][0], labels[i][1]) for i in crow_indices]
        # Sort by prediction value (descending)
        crow_preds.sort(key=lambda x: x[1], reverse=True)
        
        print("\nCrow species predictions (sorted by confidence):")
        for idx, pred_value, scientific_name, common_name in crow_preds:
            print(f"Index: {idx}, Value: {pred_value:.6f}, Species: {scientific_name} ({common_name})")
        
        # Check if any crow species is in the top 10
        top_crow = False
        for idx in top_indices:
            if idx in crow_indices:
                top_crow = True
                break
        
        if top_crow:
            print("\nA crow species was found in the top 10 predictions!")
        else:
            print("\nNo crow species found in the top 10 predictions.")
    else:
        print("No crow species found in the label list.")

if __name__ == "__main__":
    main()

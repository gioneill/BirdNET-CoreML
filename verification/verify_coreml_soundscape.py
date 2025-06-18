import numpy as np
import os
import sys
import coremltools
import argparse

# Handle TensorFlow import issues
try:
    import tensorflow as tf
    # Ensure we can access keras
    if not hasattr(tf, 'keras'):
        from tensorflow import keras
        tf.keras = keras
except ImportError as e:
    print(f"TensorFlow import error: {e}")
    sys.exit(1)

def load_audio_with_tf(audio_path, target_sr=48000):
    """Load audio file using TensorFlow."""
    try:
        # Try TensorFlow audio loading
        audio_binary = tf.io.read_file(audio_path)
        audio, sr = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=-1)  # Remove channel dimension if mono
        
        # Convert to numpy
        audio = audio.numpy().astype(np.float32)
        sr = sr.numpy()
        
        # Resample if needed (simple linear interpolation)
        if sr != target_sr:
            # Simple resampling using linear interpolation
            old_len = len(audio)
            new_len = int(old_len * target_sr / sr)
            indices = np.linspace(0, old_len - 1, new_len)
            audio = np.interp(indices, np.arange(old_len), audio)
            sr = target_sr
        
        return audio.astype(np.float32), sr
        
    except Exception as e:
        print(f"Could not load audio file {audio_path}: {e}")
        sys.exit(1)

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
    parser.add_argument('--audio_path', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "soundscape.wav"), help='Path to the audio file to analyze.')
    parser.add_argument('--model_path', type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "coreml_export/output/audio-model-export-corrected.mlpackage"), help='Path to the CoreML model.')
    parser.add_argument('--threshold', type=float, default=0.03, help='Probability threshold for displaying species.')
    args = parser.parse_args()

    label_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "coreml_export/input/labels/en_us.txt")
    
    print(f"Loading CoreML model from {args.model_path}")
    model = coremltools.models.MLModel(args.model_path)
    print("Model loaded.")

    # Load labels
    print(f"Loading labels from {label_path}")
    labels = load_labels(label_path)
    print(f"Loaded {len(labels)} labels")

    audio, sr = load_audio_with_tf(args.audio_path)
    print(f"Audio loaded from {args.audio_path}: {len(audio)/sr:.2f} seconds, {sr} Hz")

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

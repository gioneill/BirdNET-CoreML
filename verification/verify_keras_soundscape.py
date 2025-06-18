import numpy as np
import os
import sys
import argparse
import tensorflow as tf

def load_audio_with_tf(audio_path, target_sr=48000):
    """Load audio file using TensorFlow with fallback for 8-bit WAV files."""
    try:
        # Try TensorFlow audio loading first (works for 16-bit WAV)
        audio_binary = tf.io.read_file(audio_path)
        audio, sr = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=-1)  # Remove channel dimension if mono
        
        # Convert to numpy
        audio = audio.numpy().astype(np.float32)
        sr = sr.numpy()
        
        print(f"Loaded {audio_path} using TensorFlow (16-bit)")
        
    except Exception as e:
        if "Can only read 16-bit WAV files" in str(e):
            print(f"TensorFlow failed on 8-bit WAV, using manual WAV parsing...")
            # Manual WAV file parsing for 8-bit files
            audio, sr = load_8bit_wav_manual(audio_path)
        else:
            print(f"Could not load audio file {audio_path}: {e}")
            sys.exit(1)
    
    # Resample if needed (simple linear interpolation)
    if sr != target_sr:
        old_len = len(audio)
        new_len = int(old_len * target_sr / sr)
        indices = np.linspace(0, old_len - 1, new_len)
        audio = np.interp(indices, np.arange(old_len), audio)
        sr = target_sr
    
    return audio.astype(np.float32), sr

def load_8bit_wav_manual(audio_path):
    """Manually parse 8-bit WAV files."""
    import struct
    
    with open(audio_path, 'rb') as f:
        # Read WAV header
        riff = f.read(4)  # Should be b'RIFF'
        if riff != b'RIFF':
            raise ValueError("Not a valid WAV file")
        
        file_size = struct.unpack('<I', f.read(4))[0]
        wave = f.read(4)  # Should be b'WAVE'
        if wave != b'WAVE':
            raise ValueError("Not a valid WAV file")
        
        # Find fmt chunk
        while True:
            chunk_id = f.read(4)
            chunk_size = struct.unpack('<I', f.read(4))[0]
            
            if chunk_id == b'fmt ':
                fmt_data = f.read(chunk_size)
                audio_format = struct.unpack('<H', fmt_data[0:2])[0]
                num_channels = struct.unpack('<H', fmt_data[2:4])[0]
                sample_rate = struct.unpack('<I', fmt_data[4:8])[0]
                bits_per_sample = struct.unpack('<H', fmt_data[14:16])[0]
                break
            else:
                f.seek(chunk_size, 1)  # Skip this chunk
        
        # Find data chunk
        while True:
            chunk_id = f.read(4)
            chunk_size = struct.unpack('<I', f.read(4))[0]
            
            if chunk_id == b'data':
                break
            else:
                f.seek(chunk_size, 1)  # Skip this chunk
        
        # Read audio data
        audio_data = f.read(chunk_size)
        
        if bits_per_sample == 8:
            # 8-bit samples are unsigned (0-255), convert to signed (-1 to 1)
            audio = np.frombuffer(audio_data, dtype=np.uint8).astype(np.float32)
            audio = (audio - 128.0) / 128.0  # Convert to range [-1, 1]
        elif bits_per_sample == 16:
            # 16-bit samples are signed
            audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            audio = audio / 32768.0  # Convert to range [-1, 1]
        else:
            raise ValueError(f"Unsupported bit depth: {bits_per_sample}")
        
        # Convert to mono if stereo
        if num_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)
        
        print(f"Loaded 8-bit WAV: {len(audio)} samples, {sample_rate} Hz, {num_channels} channel(s)")
        return audio, sample_rate

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
    parser = argparse.ArgumentParser(description="Analyze a soundscape file with a Keras model.")
    parser.add_argument('--audio_path', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "soundscape.wav"), help='Path to the audio file to analyze.')
    parser.add_argument('--model_path', type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "coreml_export/input/audio-model.h5"), help='Path to the Keras model.')
    parser.add_argument('--threshold', type=float, default=0.03, help='Probability threshold for displaying species.')
    args = parser.parse_args()

    # Add the specific input directory to path
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "coreml_export", "input"))
    from MelSpecLayerSimple_fixed import MelSpecLayerSimple
    
    # Register custom layer for model loading
    try:
        tf.keras.saving.register_keras_serializable()(MelSpecLayerSimple)
    except AttributeError:
        tf.keras.utils.get_custom_objects()["MelSpecLayerSimple"] = MelSpecLayerSimple

    label_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "coreml_export/input/labels/en_us.txt")
    
    print(f"Loading Keras model from {args.model_path}")
    model = tf.keras.models.load_model(args.model_path, custom_objects={"MelSpecLayerSimple": MelSpecLayerSimple})
    print("Model loaded.")

    # Load labels
    print(f"Loading labels from {label_path}")
    labels = load_labels(label_path)
    print(f"Loaded {len(labels)} labels")

    audio, sr = load_audio_with_tf(args.audio_path)
    print(f"Audio loaded from {args.audio_path}: {len(audio)/sr:.2f} seconds, {sr} Hz")

    segment_duration = 3.0
    segment_length = int(segment_duration * sr)
    
    # Process audio in chunks
    for i in range(0, len(audio), segment_length):
        start_time = i / sr
        end_time = (i + segment_length) / sr
        print(f"\n--- Analyzing segment: {start_time:.2f}s - {end_time:.2f}s (Threshold: {args.threshold}) ---")

        chunk = audio[i:i + segment_length]
        
        if len(chunk) < segment_length:
            chunk = np.pad(chunk, (0, segment_length - len(chunk)), 'constant')
        
        # Model expects shape [1, 144000]
        input_data = np.expand_dims(chunk, axis=0).astype(np.float32)
        
        preds = model.predict(input_data)

        if preds.ndim == 2 and preds.shape[0] == 1:
            preds = preds[0]

        # Get indices and values of predictions above the threshold
        top_indices = np.where(preds >= args.threshold)[0]
        
        # Sort the predictions by value (descending)
        sorted_indices = sorted(top_indices, key=lambda i: preds[i], reverse=True)
        
        if not np.any(sorted_indices):
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

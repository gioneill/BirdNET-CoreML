import numpy as np
import librosa

try:
    import tensorflow as tf
    keras = tf.keras
except (ImportError, AttributeError):
    try:
        import keras
    except ImportError:
        raise ImportError("Neither tensorflow.keras nor standalone keras could be imported.")

def load_audio(audio_path, target_sr=48000, segment_duration=3.0):
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    segment_length = int(segment_duration * sr)
    if len(audio) < segment_length:
        audio = np.pad(audio, (0, segment_length - len(audio)), 'constant')
    else:
        audio = audio[:segment_length]
    return audio, sr

def main():
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from custom_layers import SimpleSpecLayer
    if hasattr(keras, "saving") and hasattr(keras.saving, "register_keras_serializable"):
        keras.saving.register_keras_serializable()(SimpleSpecLayer)
    else:
        keras.utils.get_custom_objects()["SimpleSpecLayer"] = SimpleSpecLayer

    model_path = "../model/BirdNET_6000_RAW_model.keras"
    audio_path = "crow.wav"
    print(f"Loading Keras model from {model_path}")
    model = keras.models.load_model(model_path, custom_objects={"SimpleSpecLayer": SimpleSpecLayer})
    print("Model loaded.")

    audio, sr = load_audio(audio_path)
    print(f"Audio loaded: {len(audio)/sr:.2f} seconds, {sr} Hz")
    print(f"Audio stats: min={np.min(audio):.6f}, max={np.max(audio):.6f}, mean={np.mean(audio):.6f}, std={np.std(audio):.6f}")

    # Model expects shape [1, 144000]
    input_data = np.expand_dims(audio, axis=0).astype(np.float32)
    print(f"Input shape: {input_data.shape}")

    preds = model.predict(input_data)
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
        print(f"{i+1}. Index: {idx}, Value: {preds[idx]:.6f}")

if __name__ == "__main__":
    main()

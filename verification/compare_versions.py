#!/usr/bin/env python3
"""
Compare outputs between original and fixed versions of MelSpecLayerSimple
and verify mathematical equivalence between Keras and CoreML models.
"""

import numpy as np
import os
import sys
import shutil
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

try:
    import coremltools
except ImportError as e:
    print(f"CoreMLTools import error: {e}")
    sys.exit(1)

def load_audio_or_generate(audio_path, target_sr=48000, duration=3):
    """Load audio file or generate synthetic audio if file doesn't exist or can't load."""
    # Try to load with TensorFlow first
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
        
        return audio.astype(np.float32)
        
    except Exception as e:
        print(f"Could not load audio file {audio_path}: {e}")
        print("Generating synthetic audio instead...")
        
        # Generate synthetic test audio
        t = np.linspace(0, duration, duration * target_sr, False)
        
        # Mix of different frequencies to create realistic-ish bird sound pattern
        audio = (0.3 * np.sin(2 * np.pi * 1000 * t) +  # 1kHz
                 0.2 * np.sin(2 * np.pi * 2500 * t) +  # 2.5kHz
                 0.1 * np.sin(2 * np.pi * 4000 * t) +  # 4kHz
                 0.05 * np.random.normal(0, 0.1, len(t)))  # Some noise
        
        # Add some amplitude modulation for realism
        modulation = 0.8 + 0.2 * np.sin(2 * np.pi * 3 * t)
        audio = audio * modulation
        
        # Normalize to [-1, 1]
        audio = audio / np.max(np.abs(audio))
        
        return audio.astype(np.float32)

def clear_keras_cache():
    """Clear Keras cache to force reloading of custom layers."""
    try:
        # Try different ways to access custom objects
        if hasattr(tf, 'keras') and hasattr(tf.keras, 'utils'):
            if hasattr(tf.keras.utils, 'get_custom_objects'):
                tf.keras.utils.get_custom_objects().clear()
        
        # Alternative import path
        from tensorflow.keras.utils import get_custom_objects
        get_custom_objects().clear()
    except (AttributeError, ImportError):
        # If we can't clear custom objects, continue anyway
        print("Warning: Could not clear Keras custom objects cache")
    
    # Clear module cache
    modules_to_remove = [mod for mod in sys.modules.keys() if 'MelSpecLayerSimple' in mod]
    for mod in modules_to_remove:
        del sys.modules[mod]

def test_keras_version(layer_file, model_path, audio_segment, version_name):
    """Test a specific version of the MelSpecLayerSimple layer."""
    print(f"\n=== Testing Keras with {version_name} ===")
    
    # Clear cache
    clear_keras_cache()
    
    # Copy the desired layer version
    layer_target = os.path.join(os.path.dirname(layer_file), "MelSpecLayerSimple.py")
    shutil.copy2(layer_file, layer_target)
    
    # Import fresh
    sys.path.insert(0, os.path.dirname(layer_target))
    
    try:
        from MelSpecLayerSimple import MelSpecLayerSimple
        
        # Load model
        model = tf.keras.models.load_model(model_path, custom_objects={"MelSpecLayerSimple": MelSpecLayerSimple})
        
        # Make prediction
        input_data = np.expand_dims(audio_segment, axis=0)
        predictions = model.predict(input_data, verbose=0)
        
        if predictions.ndim == 2 and predictions.shape[0] == 1:
            predictions = predictions[0]
        
        print(f"Predictions shape: {predictions.shape}")
        print(f"Min: {np.min(predictions):.6f}, Max: {np.max(predictions):.6f}, Mean: {np.mean(predictions):.6f}")
        print(f"Top 5 indices: {np.argsort(predictions)[-5:][::-1]}")
        print(f"Top 5 values: {np.sort(predictions)[-5:][::-1]}")
        
        return predictions
        
    except Exception as e:
        print(f"Error testing {version_name}: {e}")
        return None
    finally:
        # Clean up imports
        if 'MelSpecLayerSimple' in sys.modules:
            del sys.modules['MelSpecLayerSimple']

def test_coreml_version(model_path, audio_segment, version_name):
    """Test CoreML model."""
    print(f"\n=== Testing CoreML {version_name} ===")
    
    try:
        model = coremltools.models.MLModel(model_path)
        input_name = model.get_spec().description.input[0].name
        output_name = model.get_spec().description.output[0].name
        
        input_data = np.expand_dims(audio_segment, axis=0).astype(np.float32)
        predictions = model.predict({input_name: input_data})
        preds = predictions[output_name]
        
        if preds.ndim == 2 and preds.shape[0] == 1:
            preds = preds[0]
        
        print(f"Predictions shape: {preds.shape}")
        print(f"Min: {np.min(preds):.6f}, Max: {np.max(preds):.6f}, Mean: {np.mean(preds):.6f}")
        print(f"Top 5 indices: {np.argsort(preds)[-5:][::-1]}")
        print(f"Top 5 values: {np.sort(preds)[-5:][::-1]}")
        
        return preds
        
    except Exception as e:
        print(f"Error testing {version_name}: {e}")
        return None

def compare_predictions(pred1, pred2, name1, name2, tolerance=1e-5):
    """Compare two prediction arrays."""
    print(f"\n=== Comparing {name1} vs {name2} ===")
    
    if pred1 is None or pred2 is None:
        print("Cannot compare - one or both predictions failed")
        return False
    
    # Calculate differences
    abs_diff = np.abs(pred1 - pred2)
    rel_diff = abs_diff / (np.abs(pred1) + 1e-10)
    
    max_abs_diff = np.max(abs_diff)
    max_rel_diff = np.max(rel_diff)
    mean_abs_diff = np.mean(abs_diff)
    
    print(f"Max absolute difference: {max_abs_diff:.8f}")
    print(f"Max relative difference: {max_rel_diff:.8f}")
    print(f"Mean absolute difference: {mean_abs_diff:.8f}")
    
    # Check if they're equivalent within tolerance
    is_equivalent = max_abs_diff < tolerance
    
    if is_equivalent:
        print(f"✅ EQUIVALENT (within tolerance {tolerance})")
    else:
        print(f"❌ DIFFERENT (exceeds tolerance {tolerance})")
        
        # Show some different indices
        diff_indices = np.where(abs_diff > tolerance)[0]
        if len(diff_indices) > 0:
            print(f"Different indices (first 10): {diff_indices[:10]}")
            for i in diff_indices[:5]:
                print(f"  Index {i}: {pred1[i]:.6f} vs {pred2[i]:.6f} (diff: {abs_diff[i]:.6f})")
    
    return is_equivalent

def main():
    parser = argparse.ArgumentParser(description="Compare Keras and CoreML model versions")
    parser.add_argument('--audio_path', type=str, 
                       default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "soundscape.wav"),
                       help='Path to test audio file')
    args = parser.parse_args()
    
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    original_layer = os.path.join(base_dir, "coreml_export/input/MelSpecLayerSimple_original.py")
    fixed_layer = os.path.join(base_dir, "coreml_export/input/MelSpecLayerSimple_fixed.py")
    model_path = os.path.join(base_dir, "coreml_export/input/audio-model.h5")
    coreml_fixed = os.path.join(base_dir, "coreml_export/output/audio-model-fixed.mlpackage")
    
    # Load test audio
    print(f"Loading audio from {args.audio_path}")
    audio = load_audio_or_generate(args.audio_path)
    
    # Use first 3 seconds
    segment_length = 3 * 48000  # 3 seconds at 48kHz
    audio_segment = audio[:segment_length]
    if len(audio_segment) < segment_length:
        audio_segment = np.pad(audio_segment, (0, segment_length - len(audio_segment)), 'constant')
    
    print(f"Testing with audio segment: {len(audio_segment)} samples")
    
    # Test different versions
    pred_original = test_keras_version(original_layer, model_path, audio_segment, "Original Layer (with cast)")
    pred_fixed = test_keras_version(fixed_layer, model_path, audio_segment, "Fixed Layer (manual magnitude)")
    pred_coreml = test_coreml_version(coreml_fixed, audio_segment, "CoreML Fixed")
    
    # Compare results
    compare_predictions(pred_original, pred_fixed, "Original Keras", "Fixed Keras", tolerance=1e-5)
    compare_predictions(pred_fixed, pred_coreml, "Fixed Keras", "CoreML Fixed", tolerance=1e-3)
    compare_predictions(pred_original, pred_coreml, "Original Keras", "CoreML Fixed", tolerance=1e-3)
    
    print("\n=== Summary ===")
    print("This verification tests whether:")
    print("1. The original TensorFlow cast operation and our manual magnitude calculation are equivalent")
    print("2. The Keras and CoreML models produce similar results")
    print("3. Our fix maintains mathematical fidelity to the original model")

if __name__ == "__main__":
    main()

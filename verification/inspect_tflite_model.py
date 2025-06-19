#!/usr/bin/env python3
"""
Inspect TensorFlow Lite model signature to understand input/output requirements.
"""

import sys
import os
import tensorflow as tf
import numpy as np

def inspect_tflite_model(model_path):
    """Inspect TFLite model signature and test with sample input."""
    print(f"🔍 Inspecting TFLite model: {model_path}")
    print("=" * 60)
    
    # Load the TFLite model
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("✅ TFLite model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load TFLite model: {e}")
        return False
    
    # Get input details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\n📥 INPUT DETAILS:")
    print(f"Number of inputs: {len(input_details)}")
    for i, detail in enumerate(input_details):
        print(f"  Input {i}:")
        print(f"    Name: {detail['name']}")
        print(f"    Shape: {detail['shape']}")
        print(f"    Data Type: {detail['dtype']}")
        print(f"    Index: {detail['index']}")
        if 'quantization_parameters' in detail:
            print(f"    Quantization: {detail['quantization_parameters']}")
    
    print(f"\n📤 OUTPUT DETAILS:")
    print(f"Number of outputs: {len(output_details)}")
    for i, detail in enumerate(output_details):
        print(f"  Output {i}:")
        print(f"    Name: {detail['name']}")
        print(f"    Shape: {detail['shape']}")
        print(f"    Data Type: {detail['dtype']}")
        print(f"    Index: {detail['index']}")
        if 'quantization_parameters' in detail:
            print(f"    Quantization: {detail['quantization_parameters']}")
    
    # Analysis
    print(f"\n🔬 ANALYSIS:")
    input_shape = input_details[0]['shape']
    output_shape = output_details[0]['shape']
    
    print(f"Primary input shape: {input_shape}")
    print(f"Primary output shape: {output_shape}")
    
    # Determine input type based on shape
    if len(input_shape) == 2 and input_shape[1] == 144000:
        print("📻 Input appears to be RAW AUDIO (3 seconds at 48kHz)")
        input_type = "raw_audio"
    elif len(input_shape) == 4 and input_shape[1] in [128, 256] and input_shape[2] in [384, 512]:
        print("📊 Input appears to be SPECTROGRAM")
        input_type = "spectrogram"
    elif len(input_shape) == 3 and input_shape[1] in [128, 256] and input_shape[2] in [384, 512]:
        print("📊 Input appears to be SPECTROGRAM (no channel dimension)")
        input_type = "spectrogram"
    else:
        print(f"❓ Unknown input format - shape: {input_shape}")
        input_type = "unknown"
    
    # Test with sample input
    print(f"\n🧪 TESTING WITH SAMPLE INPUT:")
    try:
        if input_type == "raw_audio":
            # Create dummy audio data
            sample_input = np.random.randn(*input_shape).astype(input_details[0]['dtype'])
            print(f"Created sample audio: shape={sample_input.shape}, dtype={sample_input.dtype}")
        elif input_type == "spectrogram":
            # Create dummy spectrogram data
            sample_input = np.random.randn(*input_shape).astype(input_details[0]['dtype'])
            print(f"Created sample spectrogram: shape={sample_input.shape}, dtype={sample_input.dtype}")
        else:
            # Create generic dummy data
            sample_input = np.random.randn(*input_shape).astype(input_details[0]['dtype'])
            print(f"Created sample data: shape={sample_input.shape}, dtype={sample_input.dtype}")
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], sample_input)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(f"✅ Inference successful! Output shape: {output_data.shape}")
        print(f"Output dtype: {output_data.dtype}")
        print(f"Output range: [{output_data.min():.6f}, {output_data.max():.6f}]")
        
        # Check if output looks like probabilities
        if np.all(output_data >= 0) and np.all(output_data <= 1):
            print("📊 Output appears to be probabilities (values in [0,1])")
        elif np.all(output_data >= -10) and np.all(output_data <= 10):
            print("📊 Output appears to be logits (values roughly in [-10,10])")
        else:
            print("📊 Output format unclear")
            
    except Exception as e:
        print(f"❌ Test inference failed: {e}")
        return False
    
    return True, input_type

def compare_with_keras_model(keras_model_path):
    """Compare with Keras model for reference."""
    print(f"\n🔍 KERAS MODEL COMPARISON:")
    print("=" * 60)
    
    # Add custom layer support
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "coreml_export", "input"))
    from MelSpecLayerSimple_fixed import MelSpecLayerSimple
    
    try:
        tf.keras.saving.register_keras_serializable()(MelSpecLayerSimple)
    except AttributeError:
        tf.keras.utils.get_custom_objects()["MelSpecLayerSimple"] = MelSpecLayerSimple
    
    try:
        keras_model = tf.keras.models.load_model(keras_model_path, custom_objects={"MelSpecLayerSimple": MelSpecLayerSimple})
        print(f"✅ Keras model loaded successfully")
        
        print(f"Keras input shape: {keras_model.input.shape}")
        print(f"Keras output shape: {keras_model.output.shape}")
        
        # Test with sample input
        sample_audio = np.random.randn(1, 144000).astype(np.float32)
        keras_output = keras_model.predict(sample_audio, verbose=0)
        print(f"Keras test output shape: {keras_output.shape}")
        print(f"Keras test output range: [{keras_output.min():.6f}, {keras_output.max():.6f}]")
        
    except Exception as e:
        print(f"❌ Failed to load/test Keras model: {e}")

def main():
    # Default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tflite_model_path = os.path.join(os.path.dirname(script_dir), "coreml_export", "input", "audio-model.tflite")
    keras_model_path = os.path.join(os.path.dirname(script_dir), "coreml_export", "input", "audio-model.h5")
    
    if len(sys.argv) > 1:
        tflite_model_path = sys.argv[1]
    
    if not os.path.exists(tflite_model_path):
        print(f"❌ TFLite model not found: {tflite_model_path}")
        return 1
    
    # Inspect TFLite model
    success, input_type = inspect_tflite_model(tflite_model_path)
    
    if not success:
        return 1
    
    # Compare with Keras model if available
    if os.path.exists(keras_model_path):
        compare_with_keras_model(keras_model_path)
    else:
        print(f"\n⚠️  Keras model not found for comparison: {keras_model_path}")
    
    # Summary
    print(f"\n📋 SUMMARY:")
    print("=" * 60)
    print(f"TFLite model input type: {input_type}")
    
    if input_type == "raw_audio":
        print("✅ TFLite model expects raw audio input")
        print("   → No separate preprocessing needed")
        print("   → Can use audio loading pipeline directly")
    elif input_type == "spectrogram":
        print("⚠️  TFLite model expects preprocessed spectrogram input")
        print("   → Need to apply MelSpecLayerSimple preprocessing manually")
        print("   → Convert audio to spectrogram before inference")
    else:
        print("❓ Input type unclear - manual investigation needed")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

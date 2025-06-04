#!/usr/bin/env python3
"""
BirdNET CoreML Model Testing Script (Compiled Model Approach)

This script first compiles the CoreML model and then uses it for inference.
"""

import os
import sys
import numpy as np
import librosa
import subprocess
import json
import tempfile
import argparse
from pathlib import Path

def load_audio(audio_path, target_sr=48000, duration=3.0):
    """
    Load and preprocess audio file.
    
    Args:
        audio_path: Path to the audio file
        target_sr: Target sample rate (default: 48000 Hz)
        duration: Duration to extract (default: 3.0 seconds)
        
    Returns:
        Audio data as numpy array with shape (1, 144000)
    """
    print(f"Loading audio from {audio_path}...")
    try:
        # Load audio with librosa (handles resampling automatically)
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        
        # Print audio information
        print(f"Audio loaded: {len(audio)/sr:.2f} seconds, {sr} Hz")
        
        # If audio is longer than the required duration, we'll extract segments
        if len(audio) > int(duration * target_sr):
            # For this example, just take the first 3 seconds
            audio = audio[:int(duration * target_sr)]
            print(f"Extracted first {duration} seconds of audio")
        elif len(audio) < int(duration * target_sr):
            # If audio is shorter, pad with zeros
            padding = int(duration * target_sr) - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')
            print(f"Audio was shorter than {duration} seconds, padded with zeros")
        
        return audio
    except Exception as e:
        print(f"Error loading audio: {e}")
        sys.exit(1)

def save_audio_for_coreml(audio, output_path, sample_rate=48000):
    """
    Save preprocessed audio to a file that can be used by CoreML.
    
    Args:
        audio: Preprocessed audio data
        output_path: Path to save the audio file
        sample_rate: Sample rate of the audio
    """
    try:
        import soundfile as sf
        sf.write(output_path, audio, sample_rate, format='WAV')
        print(f"Saved preprocessed audio to {output_path}")
    except Exception as e:
        print(f"Error saving audio: {e}")
        sys.exit(1)

def compile_coreml_model(model_path, output_dir):
    """
    Compile the CoreML model using xcrun coremlcompiler.
    
    Args:
        model_path: Path to the CoreML model package
        output_dir: Directory to save the compiled model
        
    Returns:
        Path to the compiled model
    """
    print(f"\nCompiling CoreML model from {model_path}...")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the command to compile the model
    cmd = [
        "xcrun", "coremlcompiler", "compile", 
        model_path, 
        output_dir
    ]
    
    try:
        # Run the command and capture the output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Model compilation successful!")
        print(result.stdout)
        
        # The compiled model should be in the output directory with a .mlmodelc extension
        compiled_model_path = os.path.join(output_dir, os.path.basename(model_path).replace('.mlpackage', '.mlmodelc'))
        if not os.path.exists(compiled_model_path):
            # Try to find any .mlmodelc directory in the output directory
            for item in os.listdir(output_dir):
                if item.endswith('.mlmodelc'):
                    compiled_model_path = os.path.join(output_dir, item)
                    break
        
        if os.path.exists(compiled_model_path):
            print(f"Compiled model saved to: {compiled_model_path}")
            return compiled_model_path
        else:
            print(f"Could not find compiled model in {output_dir}")
            print(f"Directory contents: {os.listdir(output_dir)}")
            sys.exit(1)
            
    except subprocess.CalledProcessError as e:
        print(f"Error compiling model: {e}")
        print(f"Command output: {e.output}")
        print(f"Command stderr: {e.stderr}")
        
        # Try an alternative approach
        print("\nTrying alternative compilation approach...")
        
        # Try using the coremlc command
        alt_cmd = [
            "xcrun", "coremlc", "compile", 
            model_path, 
            output_dir
        ]
        
        try:
            alt_result = subprocess.run(alt_cmd, capture_output=True, text=True, check=True)
            print("Alternative compilation successful!")
            print(alt_result.stdout)
            
            # Try to find the compiled model
            for item in os.listdir(output_dir):
                if item.endswith('.mlmodelc'):
                    compiled_model_path = os.path.join(output_dir, item)
                    print(f"Compiled model saved to: {compiled_model_path}")
                    return compiled_model_path
            
            print(f"Could not find compiled model in {output_dir}")
            print(f"Directory contents: {os.listdir(output_dir)}")
            sys.exit(1)
            
        except subprocess.CalledProcessError as e2:
            print(f"Alternative compilation failed: {e2}")
            print(f"Command output: {e2.output}")
            print(f"Command stderr: {e2.stderr}")
            sys.exit(1)

def run_inference_with_compiled_model(model_path, audio_path):
    """
    Run inference using the compiled CoreML model.
    
    Args:
        model_path: Path to the compiled CoreML model
        audio_path: Path to the preprocessed audio file
        
    Returns:
        Prediction results
    """
    print(f"\nRunning inference with compiled model: {model_path}")
    
    # Let's try a Python approach using a subprocess to run a Python script
    # that uses the compiled model
    with tempfile.TemporaryDirectory() as temp_dir:
        python_script = f"""
import sys
import numpy as np
import soundfile as sf
import os

# Try to import coremltools with error handling
try:
    import coremltools as ct
    print("Successfully imported coremltools")
except ImportError as e:
    print(f"Error importing coremltools: {{e}}")
    sys.exit(1)

# Load the audio file
try:
    audio, sr = sf.read("{audio_path}")
    print(f"Audio loaded: {{len(audio)}} samples, {{sr}} Hz")
    
    # Ensure it's the right shape for the model (1, 144000)
    if len(audio.shape) > 1 and audio.shape[1] > 1:
        # Convert stereo to mono by averaging channels
        audio = np.mean(audio, axis=1)
    
    # Reshape to match model input shape (1, 144000)
    audio = audio.reshape(1, -1).astype(np.float32)
    print(f"Audio reshaped to: {{audio.shape}}")
except Exception as e:
    print(f"Error loading audio: {{e}}")
    sys.exit(1)

# Load the model
try:
    model = ct.models.MLModel("{model_path}")
    print("Model loaded successfully")
    print(f"Model inputs: {{model.input_description}}")
    print(f"Model outputs: {{model.output_description}}")
except Exception as e:
    print(f"Error loading model: {{e}}")
    sys.exit(1)

# Run inference
try:
    # Get the input name
    input_name = "input"  # Default name
    if hasattr(model.input_description, 'input_features'):
        input_name = model.input_description.input_features[0].name
    
    # Create input dictionary
    input_dict = {{input_name: audio}}
    
    # Run prediction
    results = model.predict(input_dict)
    print("\\nPrediction results:")
    print(results)
    
    # Try to extract and display top predictions
    if 'classLabel' in results:
        print(f"\\nTop predicted class: {{results['classLabel']}}")
    
    if 'classLabel_probs' in results:
        probs = results['classLabel_probs']
        if isinstance(probs, dict):
            # Sort by probability
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
            print("\\nTop 5 predictions:")
            for label, prob in sorted_probs:
                print(f"{{label}}: {{prob:.4f}}")
        elif hasattr(probs, 'shape'):
            # If it's an array, get top 5 indices
            top_indices = np.argsort(probs)[-5:][::-1]
            print("\\nTop 5 predictions (indices):")
            for i, idx in enumerate(top_indices):
                print(f"Class {{idx}}: {{probs[idx]:.4f}}")
except Exception as e:
    print(f"Error during inference: {{e}}")
    sys.exit(1)
        """
        
        # Save the Python script to a temporary file
        python_file = os.path.join(temp_dir, "run_model.py")
        with open(python_file, "w") as f:
            f.write(python_script)
        
        # Run the Python script
        python_cmd = ["python3", python_file]
        try:
            result = subprocess.run(python_cmd, capture_output=True, text=True, check=True)
            print("Inference successful!")
            print("\nResults:")
            print(result.stdout)
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error running inference: {e}")
            print(f"Command output: {e.output}")
            print(f"Command stderr: {e.stderr}")
            
            # If Python approach fails, try a simpler approach with Swift
            print("\nTrying alternative approach with Swift...")
            
            swift_script = f"""
            import Foundation
            import CoreML

            // Load the model
            let modelURL = URL(fileURLWithPath: "{model_path}")
            
            do {{
                let model = try MLModel(contentsOf: modelURL)
                print("Model loaded successfully")
                
                // Print model information
                print("Model description: \\(model.modelDescription)")
                
                // Print input descriptions
                print("\\nInput descriptions:")
                for (name, desc) in model.modelDescription.inputDescriptionsByName {{
                    print("\\(name): \\(desc)")
                }}
                
                // Print output descriptions
                print("\\nOutput descriptions:")
                for (name, desc) in model.modelDescription.outputDescriptionsByName {{
                    print("\\(name): \\(desc)")
                }}
                
            }} catch {{
                print("Error: \\(error)")
            }}
            """
            
            # Save the Swift script to a temporary file
            swift_file = os.path.join(temp_dir, "inspect_model.swift")
            with open(swift_file, "w") as f:
                f.write(swift_script)
            
            # Run the Swift script
            swift_cmd = ["swift", swift_file]
            try:
                swift_result = subprocess.run(swift_cmd, capture_output=True, text=True, check=True)
                print("Model inspection successful!")
                print("\nModel information:")
                print(swift_result.stdout)
                
                # Now try to use the command-line tool to run the model
                print("\nTrying to use coremlcompiler to run the model...")
                run_cmd = [
                    "xcrun", "coremlcompiler", "run",
                    model_path,
                    audio_path
                ]
                
                try:
                    run_result = subprocess.run(run_cmd, capture_output=True, text=True, check=True)
                    print("Command-line inference successful!")
                    print("\nResults:")
                    print(run_result.stdout)
                    return run_result.stdout
                except subprocess.CalledProcessError as e3:
                    print(f"Command-line inference failed: {e3}")
                    print(f"Command output: {e3.output}")
                    print(f"Command stderr: {e3.stderr}")
                    return "Inference failed with all methods. Please check the model and input format."
                
            except subprocess.CalledProcessError as e2:
                print(f"Model inspection failed: {e2}")
                print(f"Command output: {e2.output}")
                print(f"Command stderr: {e2.stderr}")
                return "Inference failed. Please check the model and input format."

def main():
    parser = argparse.ArgumentParser(description='Test BirdNET CoreML model using compiled model')
    parser.add_argument('--model', default='/Users/grego/Desktop/BirdNET_6000_RAW_WITHLABELS.mlpackage',
                        help='Path to the CoreML model package')
    parser.add_argument('--audio', default='/Users/grego/Developer/BirdNET/BirdNET-Analyzer/birdnet_analyzer/example/soundscape.wav',
                        help='Path to the audio file')
    parser.add_argument('--duration', type=float, default=3.0,
                        help='Duration of audio to process (in seconds)')
    parser.add_argument('--sample_rate', type=int, default=48000,
                        help='Target sample rate for audio processing')
    parser.add_argument('--output_dir', default='./compiled_model',
                        help='Directory to save the compiled model')
    
    args = parser.parse_args()
    
    # Load and preprocess the audio
    audio = load_audio(args.audio, args.sample_rate, args.duration)
    
    # Save the preprocessed audio to a temporary file
    temp_audio_file = "temp_audio.wav"
    save_audio_for_coreml(audio, temp_audio_file, args.sample_rate)
    
    # Compile the CoreML model
    compiled_model_path = compile_coreml_model(args.model, args.output_dir)
    
    # Run inference with the compiled model
    results = run_inference_with_compiled_model(compiled_model_path, temp_audio_file)
    
    # Clean up temporary file
    if os.path.exists(temp_audio_file):
        os.remove(temp_audio_file)
        print(f"Removed temporary file: {temp_audio_file}")
    
    print("\nTesting complete!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
BirdNET CoreML Model Testing Script (Native Approach)

This script uses a combination of Python for audio preprocessing
and Apple's native tools for CoreML inference.
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

def run_coreml_prediction(model_path, audio_path):
    """
    Run CoreML prediction using Apple's native tools.
    
    Args:
        model_path: Path to the CoreML model package
        audio_path: Path to the preprocessed audio file
        
    Returns:
        Prediction results
    """
    print("\nRunning CoreML prediction using native tools...")
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Construct the command to run the CoreML model
        cmd = [
            "xcrun", "coremlc", "predict", 
            model_path, 
            audio_path
        ]
        
        try:
            # Run the command and capture the output
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("Prediction successful!")
            print("\nRaw output:")
            print(result.stdout)
            
            # Try to parse the output as JSON
            try:
                predictions = json.loads(result.stdout)
                return predictions
            except json.JSONDecodeError:
                # If not JSON, just return the raw output
                return result.stdout
                
        except subprocess.CalledProcessError as e:
            print(f"Error running CoreML prediction: {e}")
            print(f"Command output: {e.output}")
            print(f"Command stderr: {e.stderr}")
            
            # Try an alternative approach using the Swift command-line tool
            print("\nTrying alternative approach with Swift...")
            
            # Create a Swift script to run the model
            swift_script = f"""
            import Foundation
            import CoreML

            // Load the model
            let modelURL = URL(fileURLWithPath: "{model_path}")
            let model = try! MLModel(contentsOf: modelURL)

            // Load the audio file
            let audioURL = URL(fileURLWithPath: "{audio_path}")
            let audioData = try! Data(contentsOf: audioURL)

            // Create input dictionary
            let input = try! MLDictionaryFeatureProvider(dictionary: [
                "input": audioData
            ])

            // Make prediction
            let output = try! model.prediction(from: input)

            // Print the results
            print(output)
            """
            
            # Save the Swift script to a temporary file
            swift_file = os.path.join(temp_dir, "run_model.swift")
            with open(swift_file, "w") as f:
                f.write(swift_script)
            
            # Run the Swift script
            swift_cmd = ["swift", swift_file]
            try:
                swift_result = subprocess.run(swift_cmd, capture_output=True, text=True, check=True)
                print("Swift prediction successful!")
                print("\nRaw output:")
                print(swift_result.stdout)
                return swift_result.stdout
            except subprocess.CalledProcessError as e2:
                print(f"Error running Swift prediction: {e2}")
                print(f"Command output: {e2.output}")
                print(f"Command stderr: {e2.stderr}")
                sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Test BirdNET CoreML model using native tools')
    parser.add_argument('--model', default='/Users/grego/Desktop/BirdNET_6000_RAW_WITHLABELS.mlpackage',
                        help='Path to the CoreML model package')
    parser.add_argument('--audio', default='/Users/grego/Developer/BirdNET/BirdNET-Analyzer/birdnet_analyzer/example/soundscape.wav',
                        help='Path to the audio file')
    parser.add_argument('--duration', type=float, default=3.0,
                        help='Duration of audio to process (in seconds)')
    parser.add_argument('--sample_rate', type=int, default=48000,
                        help='Target sample rate for audio processing')
    
    args = parser.parse_args()
    
    # Load and preprocess the audio
    audio = load_audio(args.audio, args.sample_rate, args.duration)
    
    # Save the preprocessed audio to a temporary file
    temp_audio_file = "temp_audio.wav"
    save_audio_for_coreml(audio, temp_audio_file, args.sample_rate)
    
    # Run CoreML prediction
    results = run_coreml_prediction(args.model, temp_audio_file)
    
    # Display results
    print("\nPrediction Results:")
    print(results)
    
    # Clean up temporary file
    if os.path.exists(temp_audio_file):
        os.remove(temp_audio_file)
        print(f"Removed temporary file: {temp_audio_file}")
    
    print("\nTesting complete!")

if __name__ == "__main__":
    main()

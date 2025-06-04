#!/usr/bin/env python3
"""
BirdNET CoreML Model Testing Script

This script loads a CoreML model for bird species classification,
processes audio input, and displays the top predictions.
"""

import os
import sys
import numpy as np
import librosa
import coremltools as ct
from scipy.io import wavfile
import argparse

def load_model(model_path):
    """Load the CoreML model."""
    print(f"Loading model from {model_path}...")
    try:
        model = ct.models.MLModel(model_path)
        print("Model loaded successfully!")
        
        # Print model metadata
        print("\nModel Metadata:")
        print(f"- Input: {model.input_description}")
        print(f"- Output: {model.output_description}")
        
        # Check if class labels are included in the model
        if hasattr(model, 'user_defined_metadata') and model.user_defined_metadata:
            print("\nModel has user-defined metadata:")
            for key, value in model.user_defined_metadata.items():
                print(f"- {key}: {value}")
        
        # Check if class labels are included in the output description
        if hasattr(model.output_description, 'classes'):
            print(f"\nFound {len(model.output_description.classes)} class labels in the model")
            print(f"First 5 classes: {model.output_description.classes[:5]}")
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

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
        
        # Reshape to match model input shape (1, 144000)
        audio = audio.reshape(1, -1)
        
        return audio
    except Exception as e:
        print(f"Error loading audio: {e}")
        sys.exit(1)

def run_inference(model, audio):
    """
    Run inference on the audio using the CoreML model.
    
    Args:
        model: Loaded CoreML model
        audio: Preprocessed audio data
        
    Returns:
        Prediction results
    """
    print("\nRunning inference...")
    try:
        # For CoreML models, the input name is often 'input'
        # Let's try to get it from the model description or use a default
        if hasattr(model.input_description, 'input_features'):
            # Try to get the name from input features
            input_name = model.input_description.input_features[0].name
        else:
            # Default to 'input' which is common in CoreML models
            input_name = 'input'
            
        print(f"Using input name: {input_name}")
        
        # Create input dictionary
        input_dict = {input_name: audio}
        
        # Run inference
        results = model.predict(input_dict)
        
        return results
    except Exception as e:
        print(f"Error during inference: {e}")
        print("Let's try with a simple approach...")
        try:
            # Try with a simple approach - just pass the audio array directly
            results = model.predict({'input': audio})
            return results
        except Exception as e2:
            print(f"Second attempt failed: {e2}")
            sys.exit(1)

def display_results(results, model, top_k=5):
    """
    Display the top k prediction results.
    
    Args:
        results: Prediction results from the model
        model: The CoreML model (to access class labels)
        top_k: Number of top predictions to display
    """
    print("\nTop predictions:")
    print(f"Results keys: {list(results.keys())}")
    
    # CoreML models can have different output structures
    # Let's handle different possibilities
    
    # Check if 'classLabel_probs' is in the results (as seen in the model metadata)
    if 'classLabel_probs' in results:
        scores = results['classLabel_probs']
        print(f"Found probability scores with shape: {scores.shape if hasattr(scores, 'shape') else 'unknown'}")
        
        # If we have a direct class label output
        if 'classLabel' in results:
            print(f"Top predicted class: {results['classLabel']}")
        
        # Try to get class labels from the model
        class_labels = None
        if hasattr(model.output_description, 'classes'):
            class_labels = model.output_description.classes
        
        # If scores is a dictionary (mapping class names to probabilities)
        if isinstance(scores, dict):
            # Sort items by probability
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            for i, (label, prob) in enumerate(sorted_items):
                print(f"{i+1}. {label}: {prob:.4f}")
        
        # If scores is a numpy array or similar
        elif hasattr(scores, '__len__'):
            # Get indices of top k predictions
            if len(scores) > 0:
                try:
                    top_indices = np.argsort(scores)[-top_k:][::-1]
                    
                    # Display top k predictions
                    for i, idx in enumerate(top_indices):
                        label = class_labels[idx] if class_labels else f"Class {idx}"
                        print(f"{i+1}. {label}: {scores[idx]:.4f}")
                except Exception as e:
                    print(f"Error sorting scores: {e}")
                    print("Raw scores:", scores)
    else:
        # Try to find any array in the results that could be scores
        for key, value in results.items():
            if hasattr(value, '__len__') and len(value) > 1:
                print(f"Using '{key}' as scores")
                scores = value
                
                try:
                    # Get indices of top k predictions
                    top_indices = np.argsort(scores)[-top_k:][::-1]
                    
                    # Display top k predictions
                    for i, idx in enumerate(top_indices):
                        print(f"{i+1}. Class {idx}: {scores[idx]:.4f}")
                except Exception as e:
                    print(f"Error processing '{key}': {e}")
                    print("Raw value:", value)
                
                break
        else:
            print("Could not find suitable prediction scores in the results")
            print("Raw results:", results)

def main():
    parser = argparse.ArgumentParser(description='Test BirdNET CoreML model')
    parser.add_argument('--model', default='/Users/grego/Desktop/BirdNET_6000_RAW_WITHLABELS.mlpackage',
                        help='Path to the CoreML model package')
    parser.add_argument('--audio', default='/Users/grego/Developer/BirdNET/BirdNET-Analyzer/birdnet_analyzer/example/soundscape.wav',
                        help='Path to the audio file')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top predictions to display')
    parser.add_argument('--duration', type=float, default=3.0,
                        help='Duration of audio to process (in seconds)')
    parser.add_argument('--sample_rate', type=int, default=48000,
                        help='Target sample rate for audio processing')
    
    args = parser.parse_args()
    
    # Load the model
    model = load_model(args.model)
    
    # Load and preprocess the audio
    audio = load_audio(args.audio, args.sample_rate, args.duration)
    
    # Run inference
    results = run_inference(model, audio)
    
    # Display results
    display_results(results, model, args.top_k)
    
    print("\nTesting complete!")

if __name__ == "__main__":
    main()

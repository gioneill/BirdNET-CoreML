#!/usr/bin/env python3
"""
Simple BirdNET CoreML Testing Script

This script uses a subprocess approach to run the CoreML model
through Apple's native tools, avoiding compatibility issues.
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

def load_audio(audio_path, target_sr=48000):
    """
    Load audio file without truncating.
    
    Args:
        audio_path: Path to the audio file
        target_sr: Target sample rate (default: 48000 Hz)
        
    Returns:
        Audio data as numpy array and sample rate
    """
    print(f"Loading audio from {audio_path}...")
    try:
        # Load audio with librosa (handles resampling automatically)
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        
        # Print audio information
        print(f"Audio loaded: {len(audio)/sr:.2f} seconds, {sr} Hz")
        
        return audio, sr
    except Exception as e:
        print(f"Error loading audio: {e}")
        sys.exit(1)

def segment_audio(audio, sr, segment_duration=3.0):
    """
    Segment audio into fixed-duration chunks.
    
    Args:
        audio: Audio data as numpy array
        sr: Sample rate
        segment_duration: Duration of each segment in seconds (default: 3.0)
        
    Returns:
        List of audio segments as numpy arrays
    """
    segment_length = int(segment_duration * sr)
    total_samples = len(audio)
    segments = []
    
    # Calculate number of full segments
    num_segments = total_samples // segment_length
    
    # Add one more segment if there's a partial segment at the end
    if total_samples % segment_length > 0:
        num_segments += 1
    
    print(f"Segmenting audio into {num_segments} segments of {segment_duration} seconds each")
    
    for i in range(num_segments):
        start = i * segment_length
        end = min(start + segment_length, total_samples)
        segment = audio[start:end]
        
        # If the last segment is shorter than segment_length, pad with zeros
        if len(segment) < segment_length:
            padding = segment_length - len(segment)
            segment = np.pad(segment, (0, padding), 'constant')
            print(f"Segment {i+1} was shorter than {segment_duration} seconds, padded with zeros")
        
        segments.append(segment)
    
    return segments

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

def compile_model(model_path):
    """
    Compile the CoreML model once and return the compiled model path.
    
    Args:
        model_path: Path to the CoreML model
        
    Returns:
        Path to the compiled model
    """
    # Create a temporary Swift script to compile the model
    with tempfile.NamedTemporaryFile(suffix='.swift', mode='w', delete=False) as f:
        swift_code = '''
import Foundation
import CoreML

do {
    // Load the model
    let modelURL = URL(fileURLWithPath: "MODEL_PATH_PLACEHOLDER")
    
    // Compile the model
    print("Compiling model...")
    let compiledModelURL = try MLModel.compileModel(at: modelURL)
    print("Model compiled successfully at: \\(compiledModelURL.path)")
    
    // Print the path to the compiled model
    print("COMPILED_MODEL_PATH:\\(compiledModelURL.path)")
} catch {
    print("Error: \\(error)")
    exit(1)
}
'''
        # Replace placeholder with actual path
        swift_code = swift_code.replace("MODEL_PATH_PLACEHOLDER", model_path)
        f.write(swift_code)
        compile_script_path = f.name
    
    # Run the Swift script to compile the model
    try:
        result = subprocess.run(["swift", compile_script_path], 
                               capture_output=True, 
                               text=True, 
                               check=True)
        
        # Extract the compiled model path from the output
        for line in result.stdout.split('\n'):
            if line.startswith("COMPILED_MODEL_PATH:"):
                compiled_model_path = line.replace("COMPILED_MODEL_PATH:", "")
                print(f"Model compiled successfully at: {compiled_model_path}")
                
                # Clean up the temporary script
                os.remove(compile_script_path)
                
                return compiled_model_path
        
    except subprocess.CalledProcessError as e:
        print(f"Error compiling model: {e}")
        print(f"Command output: {e.output}")
        print(f"Command stderr: {e.stderr}")
        
        # Clean up the temporary script
        if os.path.exists(compile_script_path):
            os.remove(compile_script_path)
        
        sys.exit(1)
    
    # If we get here, something went wrong
    print("Failed to extract compiled model path")
    sys.exit(1)

def create_swift_script(compiled_model_path, audio_path, output_path):
    """
    Create a Swift script to run the CoreML model.
    
    Args:
        compiled_model_path: Path to the compiled CoreML model
        audio_path: Path to the audio file
        output_path: Path to save the Swift script
    """
    # Create the Swift script content
    swift_code = '''
import Foundation
import CoreML
import AVFoundation

// Function to load audio file and convert to required format
func loadAudio(from url: URL) -> [Float]? {
    do {
        let audioFile = try AVAudioFile(forReading: url)
        let format = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 1)!
        let frameCount = AVAudioFrameCount(48000 * 3) // 3 seconds at 48kHz
        
        // Create a PCM buffer
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            print("Failed to create PCM buffer")
            return nil
        }
        
        // Read the audio file into the buffer
        try audioFile.read(into: buffer)
        
        // Convert buffer to array of floats
        guard let floatChannelData = buffer.floatChannelData else {
            print("Failed to get float channel data")
            return nil
        }
        
        let channelData = floatChannelData[0]
        var audioArray = [Float](repeating: 0, count: Int(frameCount))
        
        // Copy data from buffer to array
        for i in 0..<Int(buffer.frameLength) {
            audioArray[i] = channelData[i]
        }

        // ─── Immediately after "audioArray[i] = channelData[i]" ───
        let maxAmp: Float = audioArray.max() ?? 0
        let minAmp: Float = audioArray.min() ?? 0
        let meanAmp: Float = audioArray.reduce(0, +) / Float(audioArray.count)
        print(String(format: "→ Audio stats (pre‐norm): min = %.3f, max = %.3f, mean = %.6f",
                    minAmp, maxAmp, meanAmp))

        // If those values are >> ±2.0, you must normalize:
        if maxAmp > 2.0 || minAmp < -2.0 {
            print("⚠️  Detected large‐scale audio samples—dividing by 32768.0.")
            for i in 0..<audioArray.count {
                audioArray[i] /= 32768.0
            }
            // Recompute stats after normalization:
            let newMax = audioArray.max() ?? 0
            let newMin = audioArray.min() ?? 0
            let newMean = audioArray.reduce(0, +) / Float(audioArray.count)
            print(String(format: "→ Audio stats (post‐norm): min = %.3f, max = %.3f, mean = %.6f",
                        newMin, newMax, newMean))
        }
        
        return audioArray
    } catch {
        print("Error loading audio: \\(error)")
        return nil
    }
}

// Main execution
do {
    // Load the pre-compiled model
    let compiledModelURL = URL(fileURLWithPath: "COMPILED_MODEL_PATH_PLACEHOLDER")
    
    // Load the compiled model
    let model = try MLModel(contentsOf: compiledModelURL)
    print("Model loaded successfully")
    
    // Print model information
    print("\\nModel Metadata:")
    let metadata = model.modelDescription.metadata
    for (key, value) in metadata {
        print("- \\(key): \\(value)")
    }
    
    // Print input description
    print("\\nInput description:")
    for (name, desc) in model.modelDescription.inputDescriptionsByName {
        print("- \\(name): \\(desc)")
    }
    
    // Print output description
    print("\\nOutput description:")
    for (name, desc) in model.modelDescription.outputDescriptionsByName {
        print("- \\(name): \\(desc)")
    }
    
    // Load audio file
    let audioURL = URL(fileURLWithPath: "AUDIO_PATH_PLACEHOLDER")
    guard let audioData = loadAudio(from: audioURL) else {
        print("Failed to load audio data")
        exit(1)
    }
    
    print("\\nAudio loaded: \\(audioData.count) samples")
    
    // Create input for the model
    let inputName = model.modelDescription.inputDescriptionsByName.first!.key
    
    // Create a multi-array with the audio data based on the model's expected input shape
    var multiArray: MLMultiArray
    
    // Check the input shape from the model description
    let inputShape = model.modelDescription.inputDescriptionsByName.first!.value.multiArrayConstraint!.shape
    
    if inputShape.count == 3 {
        // For models expecting 3D input (like .mlmodel files with shape [1, 384, 257])
        print("Creating 3D input array with shape \\(inputShape)")
        multiArray = try MLMultiArray(shape: inputShape, dataType: .float32)
        
        // Fill the multi-array with audio data
        // Note: This is a simplified approach and may need adjustment based on the specific model
        let totalElements = inputShape.reduce(1, { $0 * $1.intValue })
        for i in 0..<min(audioData.count, totalElements) {
            multiArray[i] = NSNumber(value: audioData[i])
        }
    } else {
        // For models expecting 2D input (like .mlpackage files with shape [1, 144000])
        print("Creating 2D input array with shape [1, 144000]")
        multiArray = try MLMultiArray(shape: [1, 144000], dataType: .float32)
        
        // Fill the multi-array with audio data
        for i in 0..<min(audioData.count, 144000) {
            multiArray[i] = NSNumber(value: audioData[i])
        }
    }
    
    // Create input dictionary
    let input = try MLDictionaryFeatureProvider(dictionary: [inputName: multiArray])
    
    // Run prediction
    let output = try model.prediction(from: input)
    print("\\nPrediction successful!")
    
    // Get the results
    if let classLabel = output.featureValue(for: "classLabel")?.stringValue {
        print("\\nPredicted class: \\(classLabel)")
    }
    
    if let probsDict = output.featureValue(for: "classLabel_probs")?.dictionaryValue as? [String: Double] {
        print("\\nTop 10 predictions:")
        
        // Sort by probability and get top 10
        let sortedProbs = probsDict.sorted(by: { $0.value > $1.value }).prefix(10)
        
        // Print header
        print("Rank | Species | Confidence")
        print("-----|---------|----------")
        
        // Print each prediction with rank
        for (index, (label, prob)) in sortedProbs.enumerated() {
            print("\\(index + 1). | \\(label) | \\(String(format: "%.6f", prob))")
        }
        
        // Print total number of species detected
        print("\\nTotal species with non-zero confidence: \\(probsDict.filter { $0.value > 0 }.count)")
        
        // Print confidence distribution
        let count0to01 = probsDict.filter { $0.value > 0.0 && $0.value <= 0.1 }.count
        let count01to05 = probsDict.filter { $0.value > 0.1 && $0.value <= 0.5 }.count
        let count05to09 = probsDict.filter { $0.value > 0.5 && $0.value <= 0.9 }.count
        let count09to1 = probsDict.filter { $0.value > 0.9 && $0.value <= 1.0 }.count
        
        print("\\nConfidence distribution:")
        print("Range | Count")
        print("------|------")
        print("0.0-0.1 | \\(count0to01)")
        print("0.1-0.5 | \\(count01to05)")
        print("0.5-0.9 | \\(count05to09)")
        print("0.9-1.0 | \\(count09to1)")
    }

    // --- MODIFIED: Inspect all MultiArray outputs and then process logits ---
    print("\\n\\n--- All Available MLMultiArray Outputs ---")
    var identifiedLogitOutputName: String? = nil
    
    // Based on user-provided output: "Identity" and "Identity_1" are the MLMultiArray outputs.
    // We will assume "Identity_1" is the pre-sigmoid logits.
    // The other, "Identity", is likely used for classLabel_probs by Core ML.
    
    let allOutputNames = output.featureNames // Corrected: output itself is the MLFeatureProvider
    for name in allOutputNames {
        if let multiArray = output.featureValue(for: name)?.multiArrayValue {
            print("- Found MLMultiArray output: \(name), Shape: \(multiArray.shape), Count: \(multiArray.count), DataType: \(multiArray.dataType.rawValue)")
            if name == "Identity_1" { // Tentative assumption for logits
                identifiedLogitOutputName = name
            } else if name == "Identity" {
                 // This is likely the sigmoid output, used for classLabel_probs
                 print("  (This output '\(name)' is likely the source for 'classLabel_probs')")
            }
        } else if let stringValue = output.featureValue(for: name)?.stringValue {
            print("- Found String output: \(name) = \(stringValue)")
        } else if let dictionaryValue = output.featureValue(for: name)?.dictionaryValue {
            print("- Found Dictionary output: \(name) (contains \(dictionaryValue.count) items)")
        } else {
            print("- Found output of other type: \(name)")
        }
    }

    if let logitName = identifiedLogitOutputName, let preSigmoidLogitsOutput = output.featureValue(for: logitName)?.multiArrayValue {
        print("\\n--- Pre-Sigmoid Logits Statistics (from output: \(logitName)) ---")
        var logits: [Float] = []
        // Ensure the dataType is float32 as expected, otherwise this might crash or give bad data
        if preSigmoidLogitsOutput.dataType == .float32 {
            let count = preSigmoidLogitsOutput.count
            for i in 0..<count {
                logits.append(preSigmoidLogitsOutput[i].floatValue)
            }

            if let maxLogit = logits.max() {
                print(String(format: "Max Logit: %.6f", maxLogit))
            } else {
                print("Max Logit: N/A (array was empty or contained non-comparable values)")
            }
            
            if logits.isEmpty {
                print("Mean Logit: N/A (array is empty)")
            } else {
                let sumLogits = logits.reduce(0, +)
                let meanLogit = sumLogits / Float(logits.count)
                print(String(format: "Mean Logit: %.6f", meanLogit))
            }
            print("Number of logit values processed: \\(logits.count) from MLMultiArray with total count: \(preSigmoidLogitsOutput.count)")
        } else {
            print("Error: Expected MLMultiArray for logits ('\(logitName)') to be Float32, but found \(preSigmoidLogitsOutput.dataType.rawValue). Cannot process logits.")
        }
    } else {
        print("\\n--- Pre-Sigmoid Logits Statistics ---")
        if identifiedLogitOutputName != nil {
             print("Could not retrieve MLMultiArray for presumed logit output: '\(identifiedLogitOutputName!)'. It might not be an MLMultiArray or was not found.")
        } else {
            print("Could not identify 'Identity_1' as a candidate for pre_sigmoid_logits output among the model outputs.")
            print("Please check the 'All Available MLMultiArray Outputs' list above and verify output names.")
        }
    }
    // --- END MODIFIED ---
    
} catch {
    print("Error: \\(error)")
    exit(1)
}
'''
    
    # Replace placeholders with actual paths
    swift_code = swift_code.replace("COMPILED_MODEL_PATH_PLACEHOLDER", compiled_model_path)
    swift_code = swift_code.replace("AUDIO_PATH_PLACEHOLDER", audio_path)
    
    with open(output_path, "w") as f:
        f.write(swift_code)
    
    print(f"Created Swift script at {output_path}")
    return output_path

def process_segment(segment, compiled_model_path, sample_rate, segment_index):
    """
    Process a single audio segment with the BirdNET model.
    
    Args:
        segment: Audio segment as numpy array
        compiled_model_path: Path to the compiled CoreML model
        sample_rate: Sample rate of the audio
        segment_index: Index of the segment (for logging)
        
    Returns:
        Dictionary of species and their confidence scores
    """
    print(f"\n--- Processing segment {segment_index+1} ---")
    
    # Save the segment to a temporary file
    temp_audio_file = f"temp_audio_segment_{segment_index}.wav"
    save_audio_for_coreml(segment, temp_audio_file, sample_rate)
    
    species_probs = {}
    
    # Create a temporary directory for the Swift script
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create the Swift script
        swift_script_path = os.path.join(temp_dir, "run_model.swift")
        create_swift_script(compiled_model_path, temp_audio_file, swift_script_path)
        
        # Run the Swift script
        print(f"Running Swift script for segment {segment_index+1}...")
        try:
            # Use subprocess.Popen to get real-time output
            process = subprocess.Popen(["swift", swift_script_path], 
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      text=True,
                                      bufsize=1)
            
            # Capture output in real-time
            stdout_lines = []
            for line in iter(process.stdout.readline, ''):
                print(line, end='')  # Print immediately
                stdout_lines.append(line)
            
            # Wait for process to complete
            process.wait()
            
            # Check if process completed successfully
            if process.returncode != 0:
                stderr = process.stderr.read()
                raise subprocess.CalledProcessError(process.returncode, ["swift", swift_script_path], 
                                                   output=''.join(stdout_lines), stderr=stderr)
            
            # Extract species probabilities from the output
            output_lines = stdout_lines
            
            # Find where the top predictions start
            start_idx = -1
            for i, line in enumerate(output_lines):
                if "Top 10 predictions:" in line:
                    start_idx = i + 3  # Skip the header and separator lines
                    break
            
            # Extract species and probabilities
            if start_idx > 0:
                for i in range(start_idx, len(output_lines)):
                    line = output_lines[i].strip()
                    if not line or "Total species" in line:
                        break
                    
                    parts = line.split('|')
                    if len(parts) >= 3:
                        species = parts[1].strip()
                        prob = float(parts[2].strip())
                        species_probs[species] = prob
            
        except subprocess.CalledProcessError as e:
            print(f"Error running Swift script: {e}")
            print(f"Command output: {e.output}")
            print(f"Command stderr: {e.stderr}")
    
    # Clean up temporary file
    if os.path.exists(temp_audio_file):
        os.remove(temp_audio_file)
    
    return species_probs

def main():
    parser = argparse.ArgumentParser(description='Test BirdNET CoreML model using Swift')
    parser.add_argument('--model', default='/Users/grego/Developer/BirdNET/BirdNET-CoreML/model/BirdNET_6000_RAW_with_logits.mlpackage',
                        help='Path to the CoreML model (either .mlpackage or .mlmodel)')
    parser.add_argument('--audio', default='/Users/grego/Developer/BirdNET/verification/soundscape.wav',
                        help='Path to the audio file')
    parser.add_argument('--segment_duration', type=float, default=3.0,
                        help='Duration of each audio segment (in seconds)')
    parser.add_argument('--sample_rate', type=int, default=48000,
                        help='Target sample rate for audio processing')
    parser.add_argument('--confidence_threshold', type=float, default=0.0,
                        help='Minimum confidence threshold for including species in results')
    
    args = parser.parse_args()
    
    # Load the entire audio file
    audio, sr = load_audio(args.audio, args.sample_rate)
    
    # Segment the audio into 3-second chunks
    segments = segment_audio(audio, sr, args.segment_duration)
    
    # Compile the model once
    print("\n--- Compiling the CoreML model (this will be done only once) ---")
    compiled_model_path = compile_model(args.model)
    
    # Process each segment and collect results
    all_species = {}
    segment_detections = []
    
    for i, segment in enumerate(segments):
        print(f"\nProcessing segment {i+1} of {len(segments)}")
        species_probs = process_segment(segment, compiled_model_path, sr, i)
        
        # Store segment results
        segment_start_time = i * args.segment_duration
        segment_end_time = min((i + 1) * args.segment_duration, len(audio) / sr)
        
        segment_results = {
            "segment_index": i,
            "start_time": f"{segment_start_time:.2f}s",
            "end_time": f"{segment_end_time:.2f}s",
            "detections": []
        }
        
        # Add detections to segment results
        for species, prob in species_probs.items():
            if prob >= args.confidence_threshold:
                segment_results["detections"].append({
                    "species": species,
                    "confidence": prob
                })
                
                # Update the overall species dictionary
                if species in all_species:
                    all_species[species] = max(all_species[species], prob)
                else:
                    all_species[species] = prob
        
        segment_detections.append(segment_results)
    
    # Print overall results
    print("\n" + "="*50)
    print("OVERALL RESULTS")
    print("="*50)
    print(f"Total audio duration: {len(audio)/sr:.2f} seconds")
    print(f"Number of segments processed: {len(segments)}")
    print(f"Total unique species detected: {len(all_species)}")
    
    # Print all detected species sorted by confidence
    print("\nAll detected species (sorted by confidence):")
    print("Rank | Species | Max Confidence")
    print("-----|---------|---------------")
    
    sorted_species = sorted(all_species.items(), key=lambda x: x[1], reverse=True)
    for i, (species, prob) in enumerate(sorted_species):
        print(f"{i+1}. | {species} | {prob:.6f}")
    
    # Print segment-by-segment detections
    print("\nDetections by segment:")
    for segment in segment_detections:
        print(f"\nSegment {segment['segment_index']+1} ({segment['start_time']} - {segment['end_time']}):")
        
        if not segment['detections']:
            print("  No species detected in this segment")
            continue
            
        for detection in sorted(segment['detections'], key=lambda x: x['confidence'], reverse=True):
            print(f"  - {detection['species']}: {detection['confidence']:.6f}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()

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
        # Print waveform statistics for debugging
        print(f"Audio stats: min={np.min(audio):.6f}, max={np.max(audio):.6f}, mean={np.mean(audio):.6f}, std={np.std(audio):.6f}")
        
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
        
        // Read the audio file into the buffer, ensuring exactly 3 seconds (144000 frames)
        try audioFile.read(into: buffer, frameCount: frameCount)
        
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
        // Truncate or pad to exactly 144000 samples
        if audioArray.count > Int(frameCount) {
            audioArray = Array(audioArray.prefix(Int(frameCount)))
        } else if audioArray.count < Int(frameCount) {
            audioArray += [Float](repeating: 0, count: Int(frameCount) - audioArray.count)
        }

        // Print detailed stats for debugging
        let maxAmp: Float = audioArray.max() ?? 0
        let minAmp: Float = audioArray.min() ?? 0
        let meanAmp: Float = audioArray.reduce(0, +) / Float(audioArray.count)
        let stdAmp: Float = {
            let mean = meanAmp
            let sumSq = audioArray.reduce(0) { $0 + ($1 - mean) * ($1 - mean) }
            return sqrt(sumSq / Float(audioArray.count))
        }()
        print(String(format: "→ Audio stats (pre‐norm): min = %.3f, max = %.3f, mean = %.6f, std = %.6f",
                    minAmp, maxAmp, meanAmp, stdAmp))

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
            let newStd: Float = {
                let mean = newMean
                let sumSq = audioArray.reduce(0) { $0 + ($1 - mean) * ($1 - mean) }
                return sqrt(sumSq / Float(audioArray.count))
            }()
            print(String(format: "→ Audio stats (post‐norm): min = %.3f, max = %.3f, mean = %.6f, std = %.6f",
                        newMin, newMax, newMean, newStd))
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
    // --- Ensure normalization occurs BEFORE creating MLMultiArray and using as model input ---
    var normalizedAudioData = audioData
    let maxAmp = normalizedAudioData.max() ?? 0
    let minAmp = normalizedAudioData.min() ?? 0
    if maxAmp > 2.0 || minAmp < -2.0 {
        print("⚠️  Detected large-scale audio samples in input—dividing by 32768.0 prior to MLMultiArray conversion.")
        for i in 0..<normalizedAudioData.count {
            normalizedAudioData[i] /= 32768.0
        }
        // Print post-norm stats
        let newMax = normalizedAudioData.max() ?? 0
        let newMin = normalizedAudioData.min() ?? 0
        let newMean = normalizedAudioData.reduce(0, +) / Float(normalizedAudioData.count)
        let newStd: Float = {
            let mean = newMean
            let sumSq = normalizedAudioData.reduce(0) { $0 + ($1 - mean) * ($1 - mean) }
            return sqrt(sumSq / Float(normalizedAudioData.count))
        }()
        print(String(format: "→ Audio stats (final post‐norm for MLMultiArray): min = %.3f, max = %.3f, mean = %.6f, std = %.6f",
                    newMin, newMax, newMean, newStd))
    }

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
        for i in 0..<min(normalizedAudioData.count, totalElements) {
            multiArray[i] = NSNumber(value: normalizedAudioData[i])
        }
    } else {
        // For models expecting 2D input (like .mlpackage files with shape [1, 144000])
        print("Creating 2D input array with shape [1, 144000]")
        multiArray = try MLMultiArray(shape: [1, 144000], dataType: .float32)
        // Fill the multi-array with audio data
        for i in 0..<min(normalizedAudioData.count, 144000) {
            multiArray[i] = NSNumber(value: normalizedAudioData[i])
        }
    }
    
    // Create input dictionary
    let input = try MLDictionaryFeatureProvider(dictionary: [inputName: multiArray])
    
    // Run prediction
    let output = try model.prediction(from: input)
    print("\\nPrediction successful!")

    // --- Diagnostic: Print all output feature names and types, and show first dictionary output ---
    print("\\nDIAGNOSTIC: Output feature names and types:");
    for name in output.featureNames {
        let featureValue = output.featureValue(for: name)
        print("- \\(name): type = \\(featureValue?.type.rawValue ?? -1)")
        if featureValue?.type == .dictionary {
            if let dict = featureValue?.dictionaryValue as? [String: Double] {
                print("  First 5 keys/values for \\(name):")
                for (k, v) in dict.prefix(5) {
                    print("    \\(k): \\(v)")
                }
            }
        }
    }
    
    // Get the results: Predicted class label
    // For single output models without explicit classifier config, CoreML often names the class label output "classLabel"
    // or it might be the name of the output tensor itself if it's string type.
    // Let's try "classLabel" first, then the raw output name if that fails.
    var topClassLabel: String? = output.featureValue(for: "classLabel")?.stringValue

    if topClassLabel == nil {
        // If "classLabel" isn't found, try to find the first string output (heuristic)
        for name in output.featureNames {
            if output.featureValue(for: name)?.type == .string {
                topClassLabel = output.featureValue(for: name)?.stringValue
                print("Note: Used '\(name)' as class label output.")
                break
            }
        }
    }
    if let label = topClassLabel {
        print("\\nPredicted class: \\(label)")
    }


    // Get the results: Probabilities dictionary
    // For single output models without explicit classifier config, CoreML often names this "classLabel_probs"
    let probsOutputName = "classLabel_probs" 
    if let probsFeatureValue = output.featureValue(for: probsOutputName) {
        print("\\nInspecting output '\(probsOutputName)': Type is \(probsFeatureValue.type.rawValue)")

        if probsFeatureValue.type == .dictionary {
            if let probsDict = probsFeatureValue.dictionaryValue as? [String: Double] {
                print("Successfully cast '\(probsOutputName)' to [String: Double]. Contains \(probsDict.count) items.")
                print("\\nTop 10 predictions:")
                
                let sortedProbs = probsDict.sorted(by: { $0.value > $1.value }).prefix(10)
                
                print("Rank | Species | Confidence")
                print("-----|---------|----------")
                
                for (index, (label, prob)) in sortedProbs.enumerated() {
                    print("\\(index + 1). | \\(label) | \\(String(format: "%.6f", prob))")
                }
                
                print("\\nTotal species with non-zero confidence: \\(probsDict.filter { $0.value > 0 }.count)")
                
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
            } else {
                print("Error: '\(probsOutputName)' is a dictionary, but could not cast to [String: Double].")
            }
        } else if probsFeatureValue.type == .string {
             print("Warning: '\(probsOutputName)' is unexpectedly a String. Value: \\(probsFeatureValue.stringValue)")
        } else {
            print("Warning: '\(probsOutputName)' is not a Dictionary or String. Actual type: \(probsFeatureValue.type.rawValue)")
        }
    } else {
        print("\\nOutput '\(probsOutputName)' for probabilities dictionary not found.")
    }

    // --- Inspect all MLMultiArray Outputs (especially the raw Keras sigmoid output) ---
    print("\\n\\n--- All Available MLMultiArray Outputs (for Old-Working Model) ---")
    var rawKerasOutputName: String? = nil
    
    let allOutputNames = output.featureNames
    for name in allOutputNames {
        if let multiArrayOutput = output.featureValue(for: name)?.multiArrayValue {
            print("- Found MLMultiArray output: \(name), Shape: \(multiArrayOutput.shape), Count: \(multiArrayOutput.count), DataType: \(multiArrayOutput.dataType.rawValue)")
            // For a single output Keras model, this MLMultiArray is likely the direct Keras output
            // (e.g., from the 'sigmoid' layer), often named 'Identity' or the Keras layer name by CoreML.
            if multiArrayOutput.shape.count == 2 && multiArrayOutput.shape[0] == 1 && multiArrayOutput.shape[1] == 6522 {
                 rawKerasOutputName = name // Assume this is it
            }
        } else if output.featureValue(for: name)?.type == .string {
            // Already handled classLabel
        } else if output.featureValue(for: name)?.type == .dictionary {
            // Already handled classLabelProbs
        } else {
            print("- Found output of other type: \(name)")
        }
    }

    if let rawName = rawKerasOutputName, let rawOutputTensor = output.featureValue(for: rawName)?.multiArrayValue {
        print("\\n--- Raw Keras Output Tensor Statistics (from output: \(rawName)) ---")
        var values: [Float] = []
        if rawOutputTensor.dataType == .float32 {
            let count = rawOutputTensor.count
            for i in 0..<count {
                values.append(rawOutputTensor[i].floatValue)
            }

            if !values.isEmpty {
                let maxValue = values.max() ?? 0
                let minValue = values.min() ?? 0
                let sumValues = values.reduce(0, +)
                let meanValue = sumValues / Float(values.count)
                let stdValue = {
                    let mean = meanValue
                    let sumSq = values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) }
                    return sqrt(sumSq / Float(values.count))
                }()
                print(String(format: "Min Value: %.6f", minValue))
                print(String(format: "Max Value: %.6f", maxValue))
                print(String(format: "Mean Value: %.6f", meanValue))
                print(String(format: "Std Value: %.6f", stdValue))
            } else {
                print("No values in raw output tensor.")
            }
            print("Number of values processed: \\(values.count) from MLMultiArray with total count: \\(rawOutputTensor.count)")
            
            // Check range
            let valuesGreaterThanOne = values.filter { $0 > 1.0 }.count
            let valuesLessThanZero = values.filter { $0 < 0.0 }.count
            print("Values > 1.0: \\(valuesGreaterThanOne)")
            print("Values < 0.0: \\(valuesLessThanZero)")

        } else {
            print("Error: Expected MLMultiArray for raw Keras output ('\(rawName)') to be Float32, but found \(rawOutputTensor.dataType.rawValue).")
        }
    } else {
        print("\\n--- Raw Keras Output Tensor Statistics ---")
        if rawKerasOutputName != nil {
             print("Could not retrieve MLMultiArray for presumed raw Keras output: '\(rawKerasOutputName!)'.")
        } else {
            print("Could not identify a candidate for the raw Keras output tensor (expected shape [1, 6522]).")
            print("Please check the 'All Available MLMultiArray Outputs' list above.")
        }
    }
    
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
    # Print segment statistics for debugging
    print(f"Segment {segment_index+1} stats: min={np.min(segment):.6f}, max={np.max(segment):.6f}, mean={np.mean(segment):.6f}, std={np.std(segment):.6f}")
    
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
            
            start_idx = -1
            for i, line in enumerate(output_lines):
                if "Top 10 predictions:" in line: 
                    start_idx = i + 3 
                    break
            
            if start_idx > 0:
                for i in range(start_idx, len(output_lines)):
                    line = output_lines[i].strip()
                    if not line or "Total species" in line or "Confidence distribution:" in line:
                        break 
                    
                    parts = line.split('|')
                    if len(parts) >= 3:
                        try:
                            species = parts[1].strip()
                            prob_str = parts[2].strip()
                            if species and prob_str: 
                                prob = float(prob_str)
                                species_probs[species] = prob
                        except ValueError:
                            print(f"Warning: Could not parse probability line: {line}")
                            continue
            
        except subprocess.CalledProcessError as e:
            print(f"Error running Swift script: {e}")
            print(f"Command output: {e.output}")
            print(f"Command stderr: {e.stderr}")
    
    if os.path.exists(temp_audio_file):
        os.remove(temp_audio_file)
    
    return species_probs

def main():
    parser = argparse.ArgumentParser(description='Test BirdNET CoreML model using Swift')
    parser.add_argument('--model', default='/Users/grego/Developer/BirdNET/BirdNET-CoreML/coreml_export/output/audio-model.mlpackage',
                        help='Path to the CoreML model (either .mlpackage or .mlmodel)')
    parser.add_argument('--audio', default='verification/crow.wav',
                        help='Path to the audio file')
    parser.add_argument('--segment_duration', type=float, default=3.0,
                        help='Duration of each audio segment (in seconds)')
    parser.add_argument('--sample_rate', type=int, default=48000,
                        help='Target sample rate for audio processing')
    parser.add_argument('--confidence_threshold', type=float, default=0.0,
                        help='Minimum confidence threshold for including species in results')
    
    args = parser.parse_args()
    
    audio, sr = load_audio(args.audio, args.sample_rate)
    segments = segment_audio(audio, sr, args.segment_duration)
    
    print("\n--- Compiling the CoreML model (this will be done only once) ---")
    compiled_model_path = compile_model(args.model)
    
    all_species = {}
    segment_detections = []
    
    for i, segment in enumerate(segments):
        print(f"\nProcessing segment {i+1} of {len(segments)}")
        species_probs = process_segment(segment, compiled_model_path, sr, i)
        
        segment_start_time = i * args.segment_duration
        segment_end_time = min((i + 1) * args.segment_duration, len(audio) / sr)
        
        segment_results = {
            "segment_index": i,
            "start_time": f"{segment_start_time:.2f}s",
            "end_time": f"{segment_end_time:.2f}s",
            "detections": []
        }
        
        for species, prob in species_probs.items():
            if prob >= args.confidence_threshold:
                segment_results["detections"].append({
                    "species": species,
                    "confidence": prob
                })
                
                if species in all_species:
                    all_species[species] = max(all_species[species], prob)
                else:
                    all_species[species] = prob
        
        segment_detections.append(segment_results)
    
    print("\n" + "="*50)
    print("OVERALL RESULTS")
    print("="*50)
    print(f"Total audio duration: {len(audio)/sr:.2f} seconds")
    print(f"Number of segments processed: {len(segments)}")
    print(f"Total unique species detected: {len(all_species)}")
    
    print("\nAll detected species (sorted by confidence):")
    print("Rank | Species | Max Confidence")
    print("-----|---------|---------------")
    
    sorted_species = sorted(all_species.items(), key=lambda x: x[1], reverse=True)
    for i, (species, prob) in enumerate(sorted_species):
        print(f"{i+1}. | {species} | {prob:.6f}")

    # --- Diagnostic: Print all possible output keys and top raw values ---
    print("\n" + "="*50)
    print("DIAGNOSTIC: ALL MODEL OUTPUT KEYS AND TOP RAW VALUES")
    print("="*50)
    try:
        # Use the last processed segment's probabilities for diagnostics
        if 'species_probs' in locals() and species_probs:
            all_keys = list(species_probs.keys())
            print(f"Total output keys (species): {len(all_keys)}")
            print("First 20 output keys (species):")
            for k in all_keys[:20]:
                print(f"  - {k}")
            # Print top 20 by value
            top_items = sorted(species_probs.items(), key=lambda x: x[1], reverse=True)[:20]
            print("\nTop 20 output keys by value (for last segment):")
            for k, v in top_items:
                print(f"  - {k}: {v:.6f}")
        else:
            print("species_probs not available for diagnostics.")
    except Exception as e:
        print(f"Error during diagnostic output: {e}")
    
    print("\nDetections by segment:")
    for segment in segment_detections:
        print(f"\nSegment {segment['segment_index']+1} ({segment['start_time']} - {segment['end_time']}):")
        
        if not segment['detections']:
            print("  No species detected in this segment")
            continue
            
        for detection in sorted(segment['detections'], key=lambda x: x['confidence'], reverse=True):
            print(f"  - {detection['species']}: {detection['confidence']:.6f}")

    # --- Extended: Compare detected species to expected species list ---
    species_list_path = "/Users/grego/Developer/BirdNET/BirdNET-Analyzer/birdnet_analyzer/example/species_list.txt"
    try:
        with open(species_list_path, "r") as f:
            expected_species = set(line.strip() for line in f if line.strip())
        detected_species = set(all_species.keys())

        print("\n" + "="*50)
        print("COMPARISON TO EXPECTED SPECIES LIST")
        print("="*50)
        print(f"Species in expected list: {len(expected_species)}")
        print(f"Species detected: {len(detected_species)}")
        print(f"Expected species detected (true positives): {len(detected_species & expected_species)}")
        print(f"Expected species missed (false negatives): {len(expected_species - detected_species)}")
        print(f"Unexpected species detected (false positives): {len(detected_species - expected_species)}")

        print("\nExpected species detected:")
        for s in sorted(detected_species & expected_species):
            print(f"  - {s}")

        print("\nExpected species missed:")
        for s in sorted(expected_species - detected_species):
            print(f"  - {s}")

        print("\nUnexpected species detected:")
        for s in sorted(detected_species - expected_species):
            print(f"  - {s}")

    except Exception as e:
        print(f"\nCould not compare to expected species list: {e}")

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()

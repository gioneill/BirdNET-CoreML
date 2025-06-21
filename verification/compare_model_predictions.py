#!/usr/bin/env python3
"""
Compare predictions from two BirdNET models (Keras, TFLite, or CoreML) on a directory of audio files.
"""

import numpy as np
import os
import sys
import argparse
import csv
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import glob
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

# Handle imports based on model type
tf = None
try:
    import tensorflow as tf
    # Test basic functionality
    _ = tf.constant([1, 2, 3])
    print(f"TensorFlow {tf.__version__} loaded successfully")
except Exception as e:
    print(f"Warning: TensorFlow not available: {e}")
    tf = None

try:
    import coremltools
except ImportError as e:
    print(f"Warning: CoreML tools not available: {e}")
    coremltools = None


class ModelWrapper(ABC):
    """Abstract base class for model wrappers."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        
    @abstractmethod
    def load_model(self):
        """Load the model from the specified path."""
        pass
    
    @abstractmethod
    def predict(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Make predictions on a single audio chunk."""
        pass
    
    @abstractmethod
    def get_model_type(self) -> str:
        """Return the model type as a string."""
        pass


class KerasModel(ModelWrapper):
    """Wrapper for Keras/TensorFlow models (.h5 files)."""
    
    def load_model(self):
        """Load Keras model with custom layer support."""
        if tf is None:
            raise ImportError("TensorFlow is required for Keras models")
            
        # Add the specific input directory to path for custom layers
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "coreml_export", "input"))
        
        custom_objects = {}
        try:
            from MelSpecLayerSimple_fixed import MelSpecLayerSimple
            custom_objects["MelSpecLayerSimple"] = MelSpecLayerSimple
            # Register custom layer
            try:
                tf.keras.saving.register_keras_serializable()(MelSpecLayerSimple)
            except (AttributeError, TypeError):
                try:
                    tf.keras.utils.get_custom_objects()["MelSpecLayerSimple"] = MelSpecLayerSimple
                except AttributeError:
                    pass
        except ImportError:
            print("Warning: Could not import MelSpecLayerSimple custom layer")
            
        try:
            self.model = tf.keras.models.load_model(self.model_path, custom_objects=custom_objects)
        except AttributeError:
            # Try direct keras import
            import keras
            self.model = keras.models.load_model(self.model_path, custom_objects=custom_objects)
        
    def predict(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Make predictions using Keras model."""
        # Ensure input shape is [1, 144000]
        input_data = np.expand_dims(audio_chunk, axis=0).astype(np.float32)
        preds = self.model.predict(input_data, verbose=0)
        
        if preds.ndim == 2 and preds.shape[0] == 1:
            preds = preds[0]
            
        return preds
    
    def get_model_type(self) -> str:
        return "Keras"


class TFLiteModel(ModelWrapper):
    """Wrapper for TensorFlow Lite models (.tflite files)."""
    
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
    def load_model(self):
        """Load TFLite model and allocate tensors."""
        if tf is None:
            raise ImportError("TensorFlow is required for TFLite models")
            
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
    def predict(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Make predictions using TFLite model."""
        # Ensure input shape is [1, 144000]
        input_data = np.expand_dims(audio_chunk, axis=0).astype(np.float32)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output (logits)
        logits = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Convert logits to probabilities using softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        preds = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        if preds.ndim == 2 and preds.shape[0] == 1:
            preds = preds[0]
            
        return preds
    
    def get_model_type(self) -> str:
        return "TFLite"


class CoreMLModel(ModelWrapper):
    """Wrapper for CoreML models (.mlpackage files)."""
    
    def load_model(self):
        """Load CoreML model."""
        if coremltools is None:
            raise ImportError("coremltools is required for CoreML models")
            
        self.model = coremltools.models.MLModel(self.model_path)
        
        # Get input/output names from model spec
        spec = self.model.get_spec()
        self.input_name = spec.description.input[0].name
        self.output_name = spec.description.output[0].name
        
    def predict(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Make predictions using CoreML model."""
        # Ensure input shape is [1, 144000]
        input_data = np.expand_dims(audio_chunk, axis=0).astype(np.float32)
        
        # Make prediction
        predictions = self.model.predict({self.input_name: input_data})
        preds = predictions[self.output_name]
        
        if preds.ndim == 2 and preds.shape[0] == 1:
            preds = preds[0]
            
        return preds
    
    def get_model_type(self) -> str:
        return "CoreML"


def create_model_wrapper(model_path: str) -> ModelWrapper:
    """Factory function to create appropriate model wrapper based on file extension."""
    path = Path(model_path)
    
    if path.suffix == '.h5':
        return KerasModel(model_path)
    elif path.suffix == '.tflite':
        return TFLiteModel(model_path)
    elif path.suffix == '.mlpackage' or path.is_dir():
        return CoreMLModel(model_path)
    else:
        raise ValueError(f"Unsupported model format: {path.suffix}")


def load_audio_with_tf(audio_path: str, target_sr: int = 48000) -> Tuple[np.ndarray, int]:
    """Load audio file using TensorFlow with fallback for 8-bit WAV files."""
    if tf is None:
        raise ImportError("TensorFlow is required for audio loading")
        
    try:
        # Try TensorFlow audio loading first (works for 16-bit WAV)
        audio_binary = tf.io.read_file(audio_path)
        audio, sr = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=-1)  # Remove channel dimension if mono
        
        # Convert to numpy
        audio = audio.numpy().astype(np.float32)
        sr = sr.numpy()
        
    except Exception as e:
        if "Can only read 16-bit WAV files" in str(e):
            # Manual WAV file parsing for 8-bit files
            audio, sr = load_8bit_wav_manual(audio_path)
        else:
            raise e
    
    # Resample if needed (simple linear interpolation)
    if sr != target_sr:
        old_len = len(audio)
        new_len = int(old_len * target_sr / sr)
        indices = np.linspace(0, old_len - 1, new_len)
        audio = np.interp(indices, np.arange(old_len), audio)
        sr = target_sr
    
    return audio.astype(np.float32), sr


def load_8bit_wav_manual(audio_path: str) -> Tuple[np.ndarray, int]:
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
        
        return audio, sample_rate


def load_audio_file(audio_path: str, target_sr: int = 48000) -> Tuple[np.ndarray, int]:
    """Load audio file, supporting multiple formats."""
    ext = Path(audio_path).suffix.lower()
    
    if ext == '.wav':
        return load_audio_with_tf(audio_path, target_sr)
    elif ext == '.mp3':
        # For MP3 files, we'll use librosa if available, otherwise skip
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            return audio.astype(np.float32), sr
        except ImportError:
            # Try using pydub as a fallback
            try:
                from pydub import AudioSegment
                import tempfile
                
                # Load MP3 and convert to WAV temporarily
                audio = AudioSegment.from_mp3(audio_path)
                
                # Convert to mono if stereo
                if audio.channels > 1:
                    audio = audio.set_channels(1)
                
                # Get audio data as numpy array
                samples = np.array(audio.get_array_of_samples())
                
                # Normalize to [-1, 1]
                samples = samples.astype(np.float32) / (2**15)
                
                # Resample if needed
                if audio.frame_rate != target_sr:
                    old_len = len(samples)
                    new_len = int(old_len * target_sr / audio.frame_rate)
                    indices = np.linspace(0, old_len - 1, new_len)
                    samples = np.interp(indices, np.arange(old_len), samples)
                
                return samples, target_sr
                
            except ImportError:
                print(f"Warning: Neither librosa nor pydub available, skipping MP3 file: {audio_path}")
                return None, None
    else:
        print(f"Warning: Unsupported audio format: {ext}")
        return None, None


def load_labels(label_path: str) -> List[Tuple[str, str]]:
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


def calculate_metrics(pred1: np.ndarray, pred2: np.ndarray, top_k: int = 5) -> Dict[str, float]:
    """Calculate comparison metrics between two prediction arrays."""
    metrics = {}
    
    # Cosine similarity
    cos_sim = cosine_similarity([pred1], [pred2])[0, 0]
    metrics['cosine_similarity'] = float(cos_sim)
    
    # Mean absolute error
    mae = np.mean(np.abs(pred1 - pred2))
    metrics['mean_absolute_error'] = float(mae)
    
    # Pearson correlation
    if len(pred1) > 1:
        corr, _ = pearsonr(pred1, pred2)
        metrics['pearson_correlation'] = float(corr)
    else:
        metrics['pearson_correlation'] = 0.0
    
    # Top-K agreement
    top_k_indices1 = np.argsort(pred1)[-top_k:][::-1]
    top_k_indices2 = np.argsort(pred2)[-top_k:][::-1]
    
    # Calculate how many of the top K predictions match
    top_k_agreement = len(set(top_k_indices1) & set(top_k_indices2)) / top_k
    metrics['top_k_agreement'] = float(top_k_agreement)
    
    # Top-1 agreement
    top_1_match = int(np.argmax(pred1) == np.argmax(pred2))
    metrics['top_1_agreement'] = float(top_1_match)
    
    return metrics


def find_audio_files(directory: str, extensions: List[str] = ['.wav', '.mp3']) -> List[str]:
    """Recursively find all audio files in a directory."""
    audio_files = []
    
    for ext in extensions:
        pattern = os.path.join(directory, '**', f'*{ext}')
        files = glob.glob(pattern, recursive=True)
        audio_files.extend(files)
    
    return sorted(audio_files)


def process_audio_file(
    audio_path: str,
    model1: ModelWrapper,
    model2: ModelWrapper,
    threshold: float = 0.03,
    top_k: int = 5,
    segment_duration: float = 3.0,
    verbose: bool = False
) -> Dict:
    """Process a single audio file through both models and compare results."""
    
    # Load audio
    audio, sr = load_audio_file(audio_path)
    if audio is None:
        return None
    
    segment_length = int(segment_duration * sr)
    
    results = {
        'file': audio_path,
        'duration': len(audio) / sr,
        'segments': []
    }
    
    # Process audio in chunks
    for i in range(0, len(audio), segment_length):
        start_time = i / sr
        end_time = min((i + segment_length) / sr, len(audio) / sr)
        
        chunk = audio[i:i + segment_length]
        
        # Pad if necessary
        if len(chunk) < segment_length:
            chunk = np.pad(chunk, (0, segment_length - len(chunk)), 'constant')
        
        # Get predictions from both models
        pred1 = model1.predict(chunk)
        pred2 = model2.predict(chunk)
        
        # Calculate metrics
        metrics = calculate_metrics(pred1, pred2, top_k)
        
        segment_result = {
            'start_time': start_time,
            'end_time': end_time,
            'metrics': metrics
        }
        
        # Get top predictions from each model
        top_indices1 = np.argsort(pred1)[-top_k:][::-1]
        top_indices2 = np.argsort(pred2)[-top_k:][::-1]
        
        segment_result['model1_top_predictions'] = [(int(idx), float(pred1[idx])) for idx in top_indices1 if pred1[idx] >= threshold]
        segment_result['model2_top_predictions'] = [(int(idx), float(pred2[idx])) for idx in top_indices2 if pred2[idx] >= threshold]
        
        results['segments'].append(segment_result)
        
        if verbose:
            print(f"  Segment {start_time:.1f}-{end_time:.1f}s: "
                  f"Cosine Sim={metrics['cosine_similarity']:.3f}, "
                  f"Top-1 Match={metrics['top_1_agreement']:.0f}")
    
    return results


def aggregate_metrics(all_results: List[Dict]) -> Dict:
    """Aggregate metrics across all files and segments."""
    all_metrics = []
    
    for result in all_results:
        if result is None:
            continue
        for segment in result['segments']:
            all_metrics.append(segment['metrics'])
    
    if not all_metrics:
        return {}
    
    # Calculate mean and std for each metric
    aggregated = {}
    metric_names = all_metrics[0].keys()
    
    for metric in metric_names:
        values = [m[metric] for m in all_metrics]
        aggregated[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    return aggregated


def save_results_to_csv(results: List[Dict], output_path: str, labels: Optional[List[Tuple[str, str]]] = None):
    """Save detailed results to CSV file."""
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'file', 'start_time', 'end_time',
            'cosine_similarity', 'mean_absolute_error', 'pearson_correlation',
            'top_k_agreement', 'top_1_agreement',
            'model1_top1_species', 'model1_top1_prob',
            'model2_top1_species', 'model2_top1_prob'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            if result is None:
                continue
                
            for segment in result['segments']:
                row = {
                    'file': os.path.basename(result['file']),
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    **segment['metrics']
                }
                
                # Add top-1 predictions
                if segment['model1_top_predictions']:
                    idx, prob = segment['model1_top_predictions'][0]
                    if labels and idx < len(labels):
                        row['model1_top1_species'] = f"{labels[idx][0]} ({labels[idx][1]})"
                    else:
                        row['model1_top1_species'] = f"Species_{idx}"
                    row['model1_top1_prob'] = prob
                else:
                    row['model1_top1_species'] = 'None'
                    row['model1_top1_prob'] = 0.0
                
                if segment['model2_top_predictions']:
                    idx, prob = segment['model2_top_predictions'][0]
                    if labels and idx < len(labels):
                        row['model2_top1_species'] = f"{labels[idx][0]} ({labels[idx][1]})"
                    else:
                        row['model2_top1_species'] = f"Species_{idx}"
                    row['model2_top1_prob'] = prob
                else:
                    row['model2_top1_species'] = 'None'
                    row['model2_top1_prob'] = 0.0
                
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Compare predictions from two BirdNET models on a directory of audio files."
    )
    parser.add_argument('--model1', type=str, required=True,
                        help='Path to the first model (.h5, .tflite, or .mlpackage)')
    parser.add_argument('--model2', type=str, required=True,
                        help='Path to the second model (.h5, .tflite, or .mlpackage)')
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Directory containing audio files to analyze')
    parser.add_argument('--threshold', type=float, default=0.03,
                        help='Probability threshold for displaying species (default: 0.03)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top predictions to compare (default: 5)')
    parser.add_argument('--output_csv', type=str,
                        help='Path to save detailed results as CSV')
    parser.add_argument('--output_json', type=str,
                        help='Path to save full results as JSON')
    parser.add_argument('--label_path', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                           "coreml_export/input/labels/en_us.txt"),
                        help='Path to labels file')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress information')
    
    args = parser.parse_args()
    
    # Load labels
    print(f"Loading labels from {args.label_path}")
    try:
        labels = load_labels(args.label_path)
        print(f"Loaded {len(labels)} labels")
    except Exception as e:
        print(f"Warning: Could not load labels: {e}")
        labels = None
    
    # Create model wrappers
    print(f"\nLoading model 1: {args.model1}")
    model1 = create_model_wrapper(args.model1)
    model1.load_model()
    print(f"Model 1 loaded: {model1.get_model_type()}")
    
    print(f"\nLoading model 2: {args.model2}")
    model2 = create_model_wrapper(args.model2)
    model2.load_model()
    print(f"Model 2 loaded: {model2.get_model_type()}")
    
    # Find audio files
    print(f"\nSearching for audio files in {args.audio_dir}")
    audio_files = find_audio_files(args.audio_dir)
    print(f"Found {len(audio_files)} audio files")
    
    if not audio_files:
        print("No audio files found!")
        return
    
    # Process each audio file
    print(f"\nProcessing audio files...")
    all_results = []
    
    # Group files by directory for better organization
    from collections import defaultdict
    files_by_dir = defaultdict(list)
    for f in audio_files:
        dir_name = os.path.basename(os.path.dirname(f))
        files_by_dir[dir_name].append(f)
    
    total_processed = 0
    for dir_name, dir_files in sorted(files_by_dir.items()):
        print(f"\n{dir_name}: Processing {len(dir_files)} files...", end='', flush=True)
        dir_metrics = []
        
        for audio_file in dir_files:
            if args.verbose:
                print(f"\n  [{total_processed+1}/{len(audio_files)}] {os.path.basename(audio_file)}")
            
            result = process_audio_file(
                audio_file,
                model1,
                model2,
                threshold=args.threshold,
                top_k=args.top_k,
                verbose=args.verbose
            )
            
            if result:
                all_results.append(result)
                # Collect metrics for this directory
                for segment in result['segments']:
                    dir_metrics.append(segment['metrics'])
            
            total_processed += 1
            
            # Show progress dots for non-verbose mode
            if not args.verbose and total_processed % 5 == 0:
                print('.', end='', flush=True)
        
        # Show directory summary
        if dir_metrics:
            avg_cosine = np.mean([m['cosine_similarity'] for m in dir_metrics])
            avg_top1 = np.mean([m['top_1_agreement'] for m in dir_metrics])
            print(f" Done! (Avg cosine sim: {avg_cosine:.3f}, Top-1 agreement: {avg_top1:.1%})")
    
    # Aggregate metrics
    print(f"\n\nAggregating results...")
    aggregated = aggregate_metrics(all_results)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"COMPARISON SUMMARY: {model1.get_model_type()} vs {model2.get_model_type()}")
    print(f"{'='*60}")
    print(f"Total files processed: {len(all_results)}")
    print(f"Total segments analyzed: {sum(len(r['segments']) for r in all_results if r)}")
    print(f"\nOverall Metrics (mean ± std):")
    
    for metric, stats in aggregated.items():
        print(f"  {metric:25s}: {stats['mean']:.3f} ± {stats['std']:.3f} "
              f"(range: {stats['min']:.3f} - {stats['max']:.3f})")
    
    # Species-wise summary
    print(f"\n{'='*60}")
    print("SPECIES-WISE SUMMARY (sorted by Top-1 agreement):")
    print(f"{'='*60}")
    
    species_metrics = {}
    for dir_name in sorted(files_by_dir.keys()):
        dir_results = [r for r in all_results if any(dir_name in r['file'] for dir_name in [dir_name])]
        if dir_results:
            dir_segments = []
            for r in dir_results:
                dir_segments.extend(r['segments'])
            
            if dir_segments:
                species_metrics[dir_name] = {
                    'files': len(dir_results),
                    'segments': len(dir_segments),
                    'cosine_sim': np.mean([s['metrics']['cosine_similarity'] for s in dir_segments]),
                    'top1_agree': np.mean([s['metrics']['top_1_agreement'] for s in dir_segments])
                }
    
    # Sort by Top-1 agreement
    sorted_species = sorted(species_metrics.items(), key=lambda x: x[1]['top1_agree'], reverse=True)
    
    print(f"{'Species':30s} {'Files':>6s} {'Segments':>9s} {'Cosine':>8s} {'Top-1':>8s}")
    print("-" * 65)
    for species, metrics in sorted_species:
        print(f"{species:30s} {metrics['files']:6d} {metrics['segments']:9d} "
              f"{metrics['cosine_sim']:8.3f} {metrics['top1_agree']:7.1%}")
    
    # Save results if requested
    if args.output_csv:
        save_results_to_csv(all_results, args.output_csv, labels)
        print(f"\nDetailed results saved to: {args.output_csv}")
    
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump({
                'model1': {'path': args.model1, 'type': model1.get_model_type()},
                'model2': {'path': args.model2, 'type': model2.get_model_type()},
                'summary': aggregated,
                'detailed_results': all_results
            }, f, indent=2)
        print(f"Full results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
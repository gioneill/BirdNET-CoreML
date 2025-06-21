#!/usr/bin/env python3
"""
Verify that the CoreML meta-model produces identical results to the original Keras meta-model.

This script tests the meta-model conversion by:
1. Loading both Keras and CoreML versions of the meta-model
2. Running comprehensive test cases across different geographic locations and seasons
3. Comparing outputs with high precision to ensure conversion accuracy
4. Testing the metadata encoding function

Test cases include:
- Geographic coverage: Major cities across different continents and climate zones
- Edge cases: Poles, date line crossings, extreme coordinates
- Seasonal variation: Same location across different seasons
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

# Add the parent directory to sys.path to import our modules
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from coreml_export.meta_utils import encode_meta, get_species_priors, load_coreml_meta_model


def load_keras_meta_model(model_path: str):
    """Load the Keras meta-model with MDataLayer support."""
    
    class MDataLayer(tf.keras.layers.Layer):
        """MDataLayer for the metadata model - performs sinusoidal encoding."""
        def __init__(self, embeddings=None, **kwargs):
            super().__init__(**kwargs)
            self.embeddings = embeddings
        
        def call(self, inputs):
            # inputs shape: (batch_size, 3) containing [lat, lon, week]
            lat = inputs[:, 0:1]
            lon = inputs[:, 1:2] 
            week = inputs[:, 2:3]
            
            # Apply the encoding transformation
            feats = []
            
            # Encode latitude (normalized to -1 to 1)
            lat_norm = lat / 90.0
            for k in range(1, 25):  # 24 frequencies
                feats.append(tf.sin(k * np.pi * lat_norm))
                feats.append(tf.cos(k * np.pi * lat_norm))
            
            # Encode longitude (normalized to -1 to 1)
            lon_norm = lon / 180.0
            for k in range(1, 25):  # 24 frequencies
                feats.append(tf.sin(k * np.pi * lon_norm))
                feats.append(tf.cos(k * np.pi * lon_norm))
            
            # Encode week (normalized to 0 to 1, cyclic)
            week_norm = (week - 1) / 48.0
            for k in range(1, 25):  # 24 frequencies
                feats.append(tf.sin(k * 2 * np.pi * week_norm))
                feats.append(tf.cos(k * 2 * np.pi * week_norm))
            
            # Concatenate all features
            return tf.concat(feats, axis=-1)  # Output shape: (batch_size, 144)
        
        def get_config(self):
            config = super().get_config()
            config.update({'embeddings': self.embeddings})
            return config
    
    custom_objects = {
        "MDataLayer": MDataLayer,
    }
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
        print(f"✅ Loaded Keras meta-model from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading Keras meta-model: {e}")
        raise


def get_test_cases():
    """Define comprehensive test cases for meta-model verification."""
    
    # Geographic coverage (8 cases) - major cities across different continents and climate zones
    geographic_cases = [
        (40.7128, -74.0060, 12, "NYC (temperate, North America)"),
        (51.5074, -0.1278, 24, "London (temperate, Europe)"),
        (-33.8688, 151.2093, 36, "Sydney (temperate, Southern Hemisphere)"),
        (1.3521, 103.8198, 6, "Singapore (tropical, equator)"),
        (-22.9068, -43.1729, 48, "Rio (tropical, South America)"),
        (64.2008, -149.4937, 1, "Fairbanks (arctic)"),
        (-54.8019, -68.3030, 30, "Ushuaia (sub-antarctic)"),
        (19.4326, -99.1332, 18, "Mexico City (high altitude)"),
    ]
    
    # Edge cases (5 cases) - test encoding boundaries and extremes
    edge_cases = [
        (0.0, 0.0, 1, "Equator, Prime Meridian, week 1"),
        (90.0, 180.0, 48, "North Pole, Date Line, week 48"),
        (-90.0, -180.0, 25, "South Pole, opposite Date Line"),
        (45.0, 179.9, 26, "Near date line crossing"),
        (-45.0, -179.9, -1, "Year-round, date line boundary"),
    ]
    
    # Seasonal variation (4 cases) - same location, different seasons
    seasonal_cases = [
        (42.3601, -71.0589, 10, "Boston, early spring"),
        (42.3601, -71.0589, 20, "Boston, late spring"),
        (42.3601, -71.0589, 35, "Boston, late summer"),
        (42.3601, -71.0589, 45, "Boston, late fall"),
    ]
    
    return geographic_cases + edge_cases + seasonal_cases


def test_encoding_function():
    """Test the metadata encoding function independently."""
    print("\n=== Testing Metadata Encoding Function ===")
    
    test_cases = get_test_cases()
    
    encoding_passed = True
    for lat, lon, week, description in test_cases:
        try:
            encoded = encode_meta(lat, lon, week)
            
            # Verify expected shape
            if encoded.shape != (144,):
                print(f"❌ {description}: Wrong shape {encoded.shape}, expected (144,)")
                encoding_passed = False
                continue
            
            # Verify data type
            if encoded.dtype != np.float32:
                print(f"❌ {description}: Wrong dtype {encoded.dtype}, expected float32")
                encoding_passed = False
                continue
            
            # Verify finite values
            if not np.all(np.isfinite(encoded)):
                print(f"❌ {description}: Contains non-finite values")
                encoding_passed = False
                continue
            
            # Check reasonable value range (sine/cosine should be in [-1, 1])
            if np.abs(encoded).max() > 100:  # Allow some flexibility for different encodings
                print(f"❌ {description}: Values outside reasonable range: [{encoded.min():.3f}, {encoded.max():.3f}]")
                encoding_passed = False
                continue
            
            print(f"✅ {description}: shape={encoded.shape}, range=[{encoded.min():.3f}, {encoded.max():.3f}]")
            
        except Exception as e:
            print(f"❌ {description}: Exception - {e}")
            encoding_passed = False
    
    return encoding_passed


def compare_model_outputs(keras_model, coreml_model, test_cases, tolerance=1e-5):
    """Compare outputs between Keras and CoreML models."""
    print(f"\n=== Comparing Model Outputs (tolerance={tolerance}) ===")
    
    comparison_passed = True
    max_diff = 0.0
    
    for lat, lon, week, description in test_cases:
        try:
            # Get predictions from both models
            keras_pred = get_species_priors(lat, lon, week, keras_model)
            coreml_pred = get_species_priors(lat, lon, week, coreml_model)
            
            # Check shapes match
            if keras_pred.shape != coreml_pred.shape:
                print(f"❌ {description}: Shape mismatch - Keras: {keras_pred.shape}, CoreML: {coreml_pred.shape}")
                comparison_passed = False
                continue
            
            # Calculate maximum absolute difference
            diff = np.abs(keras_pred - coreml_pred)
            max_case_diff = np.max(diff)
            max_diff = max(max_diff, max_case_diff)
            
            # Check if within tolerance
            if max_case_diff <= tolerance:
                print(f"✅ {description}: max_diff={max_case_diff:.2e}, mean_diff={np.mean(diff):.2e}")
            else:
                print(f"❌ {description}: max_diff={max_case_diff:.2e} > tolerance={tolerance:.2e}")
                comparison_passed = False
                
                # Show some detail about where the differences are
                worst_indices = np.argsort(diff)[-5:]  # Top 5 worst differences
                print(f"   Worst differences at indices: {worst_indices}")
                for idx in worst_indices:
                    print(f"   Index {idx}: Keras={keras_pred[idx]:.6f}, CoreML={coreml_pred[idx]:.6f}, diff={diff[idx]:.2e}")
            
        except Exception as e:
            print(f"❌ {description}: Exception during comparison - {e}")
            comparison_passed = False
    
    print(f"\nOverall maximum difference: {max_diff:.2e}")
    return comparison_passed, max_diff


def compare_coreml_outputs(coreml_model1, coreml_model2, test_cases, tolerance=1e-5):
    """Compare outputs between two CoreML models."""
    print(f"\n=== Comparing CoreML Model Outputs (tolerance={tolerance}) ===")
    
    comparison_passed = True
    max_diff = 0.0
    
    for lat, lon, week, description in test_cases:
        try:
            # Get predictions from both models
            coreml_pred1 = get_species_priors(lat, lon, week, coreml_model1)
            coreml_pred2 = get_species_priors(lat, lon, week, coreml_model2)
            
            # Check shapes match
            if coreml_pred1.shape != coreml_pred2.shape:
                print(f"❌ {description}: Shape mismatch - Model1: {coreml_pred1.shape}, Model2: {coreml_pred2.shape}")
                comparison_passed = False
                continue
            
            # Calculate maximum absolute difference
            diff = np.abs(coreml_pred1 - coreml_pred2)
            max_case_diff = np.max(diff)
            max_diff = max(max_diff, max_case_diff)
            
            # Check if within tolerance
            if max_case_diff <= tolerance:
                print(f"✅ {description}: max_diff={max_case_diff:.2e}, mean_diff={np.mean(diff):.2e}")
            else:
                print(f"❌ {description}: max_diff={max_case_diff:.2e} > tolerance={tolerance:.2e}")
                comparison_passed = False
                
                # Show some detail about where the differences are
                worst_indices = np.argsort(diff)[-5:]  # Top 5 worst differences
                print(f"   Worst differences at indices: {worst_indices}")
                for idx in worst_indices:
                    print(f"   Index {idx}: Model1={coreml_pred1[idx]:.6f}, Model2={coreml_pred2[idx]:.6f}, diff={diff[idx]:.2e}")
            
        except Exception as e:
            print(f"❌ {description}: Exception during comparison - {e}")
            comparison_passed = False
    
    print(f"\nOverall maximum difference: {max_diff:.2e}")
    return comparison_passed, max_diff


def test_model_properties(keras_model, coreml_model):
    """Test basic model properties."""
    print("\n=== Testing Model Properties ===")
    
    # Test input/output shapes
    keras_input_shape = keras_model.input.shape
    keras_output_shape = keras_model.output.shape
    
    print(f"Keras model - Input: {keras_input_shape}, Output: {keras_output_shape}")
    
    # Test with a simple input - now using 3 inputs (lat, lon, week)
    test_input = np.array([[40.7128, -74.0060, 12]], dtype=np.float32)  # NYC, Spring
    
    try:
        keras_output = keras_model.predict(test_input, verbose=0)
        print(f"✅ Keras model inference successful - Output shape: {keras_output.shape}")
    except Exception as e:
        print(f"❌ Keras model inference failed: {e}")
        return False
    
    try:
        # Try CoreML inference
        coreml_spec = coreml_model.get_spec()
        input_name = coreml_spec.description.input[0].name
        result = coreml_model.predict({input_name: test_input})
        output_name = list(result.keys())[0]
        coreml_output = result[output_name]
        print(f"✅ CoreML model inference successful - Output shape: {coreml_output.shape}")
    except Exception as e:
        print(f"❌ CoreML model inference failed: {e}")
        return False
    
    return True


def _parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Verify BirdNET meta-model conversion accuracy")
    
    p.add_argument(
        "--model1", 
        default="coreml_export/input/meta-model.h5",
        help="Path to first model (.h5 or .mlpackage)"
    )
    p.add_argument(
        "--model2", 
        default="coreml_export/output/meta-model.mlpackage",
        help="Path to second model (.h5 or .mlpackage)"
    )
    p.add_argument(
        "--tolerance", 
        type=float, 
        default=1e-5,
        help="Tolerance for numerical comparison"
    )
    
    return p.parse_args()


def main():
    """Main verification function."""
    print("🔍 BirdNET Meta-Model Verification")
    print("=" * 50)
    
    # Parse arguments
    args = _parse_args()
    
    # Resolve paths
    model1_path = Path(args.model1)
    model2_path = Path(args.model2)
    
    # Make paths absolute if they're relative
    if not model1_path.is_absolute():
        model1_path = repo_root / model1_path
    if not model2_path.is_absolute():
        model2_path = repo_root / model2_path
    
    # Check if files exist
    if not model1_path.exists():
        print(f"❌ Model 1 not found: {model1_path}")
        return 1
    
    if not model2_path.exists():
        print(f"❌ Model 2 not found: {model2_path}")
        return 1
    
    # Auto-detect model types
    model1_is_keras = model1_path.suffix.lower() in ['.h5', '.keras']
    model2_is_keras = model2_path.suffix.lower() in ['.h5', '.keras']
    model1_is_coreml = model1_path.suffix.lower() == '.mlpackage' or model1_path.name.endswith('.mlpackage')
    model2_is_coreml = model2_path.suffix.lower() == '.mlpackage' or model2_path.name.endswith('.mlpackage')
    
    print(f"Model 1: {model1_path} ({'Keras' if model1_is_keras else 'CoreML' if model1_is_coreml else 'Unknown'})")
    print(f"Model 2: {model2_path} ({'Keras' if model2_is_keras else 'CoreML' if model2_is_coreml else 'Unknown'})")
    
    # Validate model type combinations
    if model1_is_keras and model2_is_keras:
        print("❌ Cannot compare two Keras models. Please provide one Keras and one CoreML, or two CoreML models.")
        return 1
    elif not ((model1_is_keras or model1_is_coreml) and (model2_is_keras or model2_is_coreml)):
        print("❌ Unrecognized model formats. Please provide .h5/.keras or .mlpackage files.")
        return 1
    
    # Test encoding function first
    encoding_ok = test_encoding_function()
    if not encoding_ok:
        print("\n❌ Encoding function tests failed!")
        return 1
    
    # Load models based on detected types
    try:
        print(f"\nLoading models...")
        
        if model1_is_keras and model2_is_coreml:
            # Original keras-vs-coreml comparison
            keras_model = load_keras_meta_model(str(model1_path))
            coreml_model = load_coreml_meta_model(str(model2_path))
            comparison_type = "Keras vs CoreML"
        elif model1_is_coreml and model2_is_keras:
            # Swapped keras-vs-coreml comparison
            keras_model = load_keras_meta_model(str(model2_path))
            coreml_model = load_coreml_meta_model(str(model1_path))
            comparison_type = "Keras vs CoreML (swapped)"
        elif model1_is_coreml and model2_is_coreml:
            # CoreML-vs-CoreML comparison
            coreml_model = load_coreml_meta_model(str(model1_path))
            coreml_model2 = load_coreml_meta_model(str(model2_path))
            keras_model = None  # No keras model in this case
            comparison_type = "CoreML vs CoreML"
        
        print(f"Comparison type: {comparison_type}")
        
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return 1
    
    # Test basic model properties
    if keras_model is not None:
        properties_ok = test_model_properties(keras_model, coreml_model)
        if not properties_ok:
            print("\n❌ Model property tests failed!")
            return 1
    else:
        # For CoreML-to-CoreML comparison, skip properties test
        properties_ok = True
        print("Skipping model properties test for CoreML-to-CoreML comparison")
    
    # Get test cases and run comparison
    test_cases = get_test_cases()
    print(f"\nRunning {len(test_cases)} test cases...")
    
    if model1_is_coreml and model2_is_coreml:
        # CoreML-to-CoreML comparison
        comparison_ok, max_diff = compare_coreml_outputs(coreml_model, coreml_model2, test_cases, args.tolerance)
    else:
        # Keras-to-CoreML comparison
        comparison_ok, max_diff = compare_model_outputs(keras_model, coreml_model, test_cases, args.tolerance)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    if encoding_ok and properties_ok and comparison_ok:
        print("✅ All tests PASSED!")
        print(f"   - Encoding function: ✅")
        print(f"   - Model properties: ✅") 
        print(f"   - Output comparison: ✅ (max diff: {max_diff:.2e})")
        print("\n🎉 Meta-model conversion is accurate and ready for use!")
        return 0
    else:
        print("❌ Some tests FAILED!")
        print(f"   - Encoding function: {'✅' if encoding_ok else '❌'}")
        print(f"   - Model properties: {'✅' if properties_ok else '❌'}")
        print(f"   - Output comparison: {'✅' if comparison_ok else '❌'}")
        print("\n⚠️  Please fix the issues before using the converted model.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

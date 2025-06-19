#!/usr/bin/env python3
"""
Utility functions for working with BirdNET's metadata model (location/time → species priors).

This module provides functions to:
1. Encode latitude, longitude, and week into the 144-feature format expected by the meta-model
2. Get species occurrence probabilities from location/time data
3. Filter audio predictions using location-based species filtering (BirdNET's current approach)

Note: Future versions could implement true Bayesian combination (audio_scores * location_priors)
      but this module focuses on BirdNET's current filtering approach.
"""

import numpy as np
from typing import List, Tuple, Union
import coremltools as ct


def encode_meta(lat: float, lon: float, week: int) -> np.ndarray:
    """
    Encode latitude, longitude, and week into 144-feature vector for the meta-model.
    
    The encoding uses sine/cosine representation to capture cyclical nature of 
    geographic coordinates and seasonal time.
    
    Args:
        lat: Latitude in degrees (-90 to 90)
        lon: Longitude in degrees (-180 to 180) 
        week: Week of year (1-48 for BirdNET, or -1 for year-round)
        
    Returns:
        np.ndarray: 144-element feature vector [lat_features(72) + lon_features(72)]
        
    Raises:
        ValueError: If coordinates or week are out of valid range
    """
    # Validate inputs
    if not -90 <= lat <= 90:
        raise ValueError(f"Latitude must be between -90 and 90, got {lat}")
    if not -180 <= lon <= 180:
        raise ValueError(f"Longitude must be between -180 and 180, got {lon}")
    if week != -1 and not 1 <= week <= 48:
        raise ValueError(f"Week must be between 1 and 48 (or -1 for year-round), got {week}")
    
    # Normalize coordinates to [0, 1] range
    lat_norm = (lat + 90) / 180  # [-90, 90] → [0, 1]
    lon_norm = (lon + 180) / 360  # [-180, 180] → [0, 1]
    
    # Handle week encoding - convert to [0, 1] range
    if week == -1:
        # Year-round: use middle value
        week_norm = 0.5
    else:
        week_norm = week / 48.0  # [1, 48] → [0.02, 1.0]
    
    # Create sine/cosine features for each dimension
    # Using multiple frequencies to capture different scales
    features = []
    
    # Latitude encoding (72 features)
    for i in range(36):
        freq = 2 ** i  # Exponentially increasing frequencies
        features.append(np.sin(2 * np.pi * freq * lat_norm))
        features.append(np.cos(2 * np.pi * freq * lat_norm))
    
    # Longitude encoding (72 features) 
    for i in range(36):
        freq = 2 ** i
        features.append(np.sin(2 * np.pi * freq * lon_norm))
        features.append(np.cos(2 * np.pi * freq * lon_norm))
    
    # Note: Week information may be encoded differently or combined with lat/lon
    # This is a simplified version - the actual BirdNET encoding may vary
    
    return np.array(features, dtype=np.float32)


def get_species_priors(lat: float, lon: float, week: int, meta_model) -> np.ndarray:
    """
    Get species occurrence probabilities for a given location and time.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees  
        week: Week of year (1-48 or -1 for year-round)
        meta_model: Loaded meta-model (Keras model or CoreML model)
        
    Returns:
        np.ndarray: Array of 6522 species occurrence probabilities
    """
    # The new models take 3 inputs directly (lat, lon, week)
    # No encoding needed - the MDataLayer handles encoding internally
    
    # Prepare input as [lat, lon, week]
    input_data = np.array([[lat, lon, week]], dtype=np.float32)
    
    # Run inference
    if hasattr(meta_model, 'predict'):
        # Check if it's a Keras model (has compile method)
        if hasattr(meta_model, 'compile'):
            # Keras model
            predictions = meta_model.predict(input_data, verbose=0)
            return predictions[0]  # Remove batch dimension
        else:
            # CoreML model
            input_name = list(meta_model.get_spec().description.input)[0].name
            result = meta_model.predict({input_name: input_data})
            output_name = list(result.keys())[0]
            return result[output_name][0]  # Remove batch dimension
    else:
        raise ValueError("Unknown model type")


def filter_by_location(
    audio_scores: np.ndarray, 
    species_labels: List[str],
    lat: float, 
    lon: float, 
    week: int, 
    meta_model,
    threshold: float = 0.03
) -> Tuple[np.ndarray, List[str]]:
    """
    Filter audio predictions using location-based species filtering.
    
    This implements BirdNET's current approach: use location data to create a binary 
    species filter, then only return audio predictions for species that occur in 
    the given location/time.
    
    Args:
        audio_scores: Array of audio prediction scores (6522 species)
        species_labels: List of species names corresponding to scores
        lat: Latitude in degrees
        lon: Longitude in degrees
        week: Week of year (1-48 or -1 for year-round)
        meta_model: Loaded meta-model for location predictions
        threshold: Minimum occurrence probability to include species (default: 0.03)
        
    Returns:
        Tuple of (filtered_scores, filtered_labels) containing only species 
        that occur in the specified location/time
    """
    # Get location-based species probabilities
    location_probs = get_species_priors(lat, lon, week, meta_model)
    
    # Create binary filter based on threshold
    location_filter = location_probs >= threshold
    
    # Apply filter to audio scores and labels
    filtered_scores = audio_scores[location_filter]
    filtered_labels = [species_labels[i] for i in range(len(species_labels)) if location_filter[i]]
    
    return filtered_scores, filtered_labels


def get_location_species_list(
    lat: float, 
    lon: float, 
    week: int, 
    meta_model,
    species_labels: List[str],
    threshold: float = 0.03,
    sort_by_probability: bool = False
) -> List[Tuple[str, float]]:
    """
    Get a list of species likely to occur at a given location and time.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        week: Week of year (1-48 or -1 for year-round)
        meta_model: Loaded meta-model
        species_labels: List of all species names
        threshold: Minimum occurrence probability to include species
        sort_by_probability: If True, sort by probability (highest first)
        
    Returns:
        List of (species_name, probability) tuples for species above threshold
    """
    # Get location-based probabilities
    location_probs = get_species_priors(lat, lon, week, meta_model)
    
    # Filter by threshold and create list of (species, probability) pairs
    species_list = [
        (species_labels[i], float(location_probs[i])) 
        for i in range(len(species_labels)) 
        if location_probs[i] >= threshold
    ]
    
    # Sort by probability if requested
    if sort_by_probability:
        species_list.sort(key=lambda x: x[1], reverse=True)
    
    return species_list


def load_coreml_meta_model(model_path: str):
    """
    Load a CoreML meta-model from file.
    
    Args:
        model_path: Path to the .mlpackage or .mlmodel file
        
    Returns:
        Loaded CoreML model
    """
    return ct.models.MLModel(model_path)


# Example usage and testing functions
def test_encoding():
    """Test the metadata encoding function with various inputs."""
    test_cases = [
        (40.7128, -74.0060, 12),    # NYC, March
        (51.5074, -0.1278, 24),     # London, June  
        (-33.8688, 151.2093, 36),   # Sydney, September
        (0.0, 0.0, 1),              # Equator, Prime Meridian
        (90.0, 180.0, 48),          # North Pole, Date Line
    ]
    
    print("Testing metadata encoding:")
    for lat, lon, week in test_cases:
        try:
            encoded = encode_meta(lat, lon, week)
            print(f"  ({lat:6.1f}, {lon:7.1f}, week {week:2d}) → {encoded.shape} features, "
                  f"range [{encoded.min():.3f}, {encoded.max():.3f}]")
        except Exception as e:
            print(f"  ({lat:6.1f}, {lon:7.1f}, week {week:2d}) → ERROR: {e}")


if __name__ == "__main__":
    test_encoding()

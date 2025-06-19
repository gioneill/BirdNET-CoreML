#!/usr/bin/env python3
"""
Inspect the meta-model.h5 to understand its structure and inputs.
"""

import h5py
import os

def inspect_h5_structure(h5_path):
    """Inspect H5 file structure"""
    print(f"Inspecting H5 file: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        print("\n=== H5 File Structure ===")
        
        def print_structure(name, obj):
            indent = "  " * (name.count('/'))
            if isinstance(obj, h5py.Group):
                print(f"{indent}{name}/ (Group)")
            elif isinstance(obj, h5py.Dataset):
                print(f"{indent}{name} (Dataset): shape={obj.shape}, dtype={obj.dtype}")
                # Print attributes
                if obj.attrs:
                    for attr_name, attr_value in obj.attrs.items():
                        print(f"{indent}  @{attr_name}: {attr_value}")
        
        f.visititems(print_structure)
        
        # Look for model config
        if 'model_config' in f.attrs:
            print(f"\n=== Model Config ===")
            import json
            try:
                config = json.loads(f.attrs['model_config'].decode('utf-8'))
                print(f"Class name: {config.get('class_name', 'Unknown')}")
                if 'config' in config:
                    model_config = config['config']
                    if 'layers' in model_config:
                        print(f"Number of layers: {len(model_config['layers'])}")
                        print("Layer info:")
                        for i, layer in enumerate(model_config['layers'][:5]):  # First 5 layers
                            layer_config = layer.get('config', {})
                            print(f"  Layer {i}: {layer.get('class_name', 'Unknown')} - {layer_config.get('name', 'unnamed')}")
                            if 'batch_input_shape' in layer_config:
                                print(f"    Input shape: {layer_config['batch_input_shape']}")
            except Exception as e:
                print(f"Error parsing model config: {e}")

# Load the model
model_path = os.path.join(os.path.dirname(__file__), "input/meta-model.h5")

try:
    inspect_h5_structure(model_path)
except Exception as e:
    print(f"Error inspecting model: {e}")

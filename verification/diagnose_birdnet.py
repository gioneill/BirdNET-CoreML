#!/usr/bin/env python3
"""
diagnose_birdnet.py

A helper script to inspect a BirdNET Core ML package, including:
  - Counting output classes
  - Checking for a final Sigmoid layer
  - (Optional) Extracting raw logits, if supported

Usage:
  python3 diagnose_birdnet.py --model /path/to/BirdNET_6522_RAW_WITHLABELS.mlpackage [--labels /path/to/labels.txt] [--logits]

Options:
  --model   Path to the .mlpackage or .mlmodel file.
  --labels  (Optional) Path to labels.txt to compare class count.
  --logits  (Optional) If set, attempts to print raw-logit statistics (requires pre_sigmoid_logits output).

Examples:
  python3 diagnose_birdnet.py --model BirdNET_6522_RAW_WITHLABELS.mlpackage --labels labels.txt
  python3 diagnose_birdnet.py --model BirdNET_with_logits.mlpackage --logits
"""

import sys
import argparse

try:
    import coremltools as ct
    import numpy as np
except ImportError:
    print("❌ Please install coremltools and numpy in your Python environment.")
    sys.exit(1)

def count_labels_in_txt(labels_path: str) -> int:
    count = 0
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count

def inspect_sigmoid_layer(model_path: str):
    spec = ct.utils.load_spec(model_path)
    if not spec.HasField("neuralNetwork"):
        print("❌ No 'neuralNetwork' section in spec.")
        return
    layers = spec.neuralNetwork.layers
    print(f"Total layers in spec: {len(layers)}\n")
    for i, layer in enumerate(layers[-5:], start=len(layers)-5):
        name = layer.name
        ltype = layer.WhichOneof("layer")
        print(f"  [{i}] Name: '{name}', Type: {ltype}")
        if ltype == "activation":
            act = layer.activation
            if act.HasField("sigmoid"):
                print("       ↳ activation = Sigmoid")
            elif act.HasField("softmax"):
                print("       ↳ activation = Softmax")
            else:
                print(f"       ↳ activation = {act.WhichOneof('NonlinearityType')}")
    last_out = layers[-1].output
    print(f"\nOutput blobs of final layer: {list(last_out)}")

def verify_model_classes(model_path: str, labels_path: str = None):
    print(f"Loading Core ML model from: {model_path}")
    m = ct.models.MLModel(model_path)
    dummy_input = np.zeros((1, 144000), dtype=np.float32)
    try:
        out = m.predict({"input": dummy_input})
    except Exception as e:
        print(f"❌ Dummy prediction failed: {e}")
        return
    if "classLabel_probs" not in out:
        print("❌ 'classLabel_probs' not in model output keys:", out.keys())
        return
    count = len(out["classLabel_probs"])
    print(f"✅ Model returns {count} classes in 'classLabel_probs'")
    if labels_path:
        txt_count = count_labels_in_txt(labels_path)
        print(f"ℹ️  '{labels_path}' has {txt_count} non-blank lines")
        if txt_count == count:
            print("✅ Label file count matches model class count.")
        else:
            print("❌ MISMATCH: model has {0} vs labels.txt {1}".format(count, txt_count))

def inspect_logits(model_path: str):
    print(f"Loading Core ML model from: {model_path}")
    m = ct.models.MLModel(model_path)
    dummy_input = np.zeros((1, 144000), dtype=np.float32)
    try:
        out = m.predict({"input": dummy_input})
    except Exception as e:
        print(f"❌ Dummy prediction failed: {e}")
        return
    if "pre_sigmoid_logits" not in out:
        print("❌ 'pre_sigmoid_logits' not found; model was not exported with logits.")
        return
    logits = out["pre_sigmoid_logits"]
    if isinstance(logits, np.ndarray):
        max_logit = np.max(logits)
        mean_logit = np.mean(logits)
        print(f"→ Raw logits stats: max = {max_logit:.3f}, mean = {mean_logit:.6f}")
    else:
        print("❌ Unexpected type for 'pre_sigmoid_logits':", type(logits))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to the Core ML package")
    parser.add_argument("--labels", help="Optional labels.txt to compare class count")
    parser.add_argument("--logits", action="store_true", help="If set, print raw-logit stats")
    args = parser.parse_args()

    if args.logits:
        inspect_sigmoid_layer(args.model)
        inspect_logits(args.model)
    else:
        inspect_sigmoid_layer(args.model)
        verify_model_classes(args.model, args.labels)

if __name__ == "__main__":
    main()

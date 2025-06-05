#!/usr/bin/env python3
"""
Export a BirdNET Raven (protobuf/frozen graph) model using BirdNET-Analyzer CLI.

Usage:
    python export_raven_model.py --tflite_model_path PATH_TO_TFLITE --output_dir OUTPUT_DIR

This script will invoke the BirdNET-Analyzer CLI to export a Raven-format model (.pb).
"""

import argparse
import subprocess
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Export BirdNET Raven model using BirdNET-Analyzer CLI")
    parser.add_argument("--tflite_model_path", type=str, required=True, help="Path to the BirdNET TFLite model (e.g., BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the exported Raven model")
    parser.add_argument("--labels", type=str, default=None, help="Path to labels.txt (optional)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct CLI command
    # The BirdNET-Analyzer CLI expects a training data folder, but for global models we may need to use a dummy folder or point to the model directly.
    # Here, we assume the CLI can be invoked to export the Raven model from the TFLite model.
    # If a custom classifier is needed, pass it via --classifier.

    cli_path = os.path.join("..", "BirdNET-Analyzer", "birdnet_analyzer", "cli.py")
    raven_out = output_dir / "BirdNET_GLOBAL_6K_V2.4_RAVEN.pb"

    cmd = [
        "python3", cli_path,
        "--model_format", "raven",
        "-c", args.tflite_model_path,
        "--output", str(raven_out)
    ]
    if args.labels:
        cmd += ["--slist", args.labels]

    print("Running BirdNET-Analyzer CLI to export Raven model:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    print(f"Raven model exported to: {raven_out}")

if __name__ == "__main__":
    main()

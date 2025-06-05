#!/usr/bin/env python3
"""
Convert a BirdNET Raven (frozen graph .pb) model to a TensorFlow SavedModel.

Usage:
    python raven_to_savedmodel.py --pb_path PATH_TO_PB --output_dir OUTPUT_DIR

This script loads a frozen graph (.pb) and exports it as a TensorFlow SavedModel.
"""

import argparse
import tensorflow as tf
import os
from pathlib import Path

def load_frozen_graph(pb_path):
    with tf.io.gfile.GFile(pb_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

def main():
    parser = argparse.ArgumentParser(description="Convert Raven .pb model to TensorFlow SavedModel")
    parser.add_argument("--pb_path", type=str, required=True, help="Path to the Raven .pb model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the TensorFlow SavedModel")
    args = parser.parse_args()

    pb_path = args.pb_path
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    graph_def = load_frozen_graph(pb_path)

    # Import the graph and export as SavedModel
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

        with tf.compat.v1.Session(graph=graph) as sess:
            # Find input and output tensors
            input_tensors = [n for n in graph_def.node if n.op == "Placeholder"]
            output_tensors = [n for n in graph_def.node if n.op in ("Softmax", "Sigmoid", "Identity")]

            if not input_tensors or not output_tensors:
                raise RuntimeError("Could not automatically find input/output tensors. Please edit the script to specify them.")

            input_name = input_tensors[0].name + ":0"
            output_name = output_tensors[-1].name + ":0"

            print(f"Using input: {input_name}")
            print(f"Using output: {output_name}")

            tf.compat.v1.saved_model.simple_save(
                sess,
                output_dir,
                inputs={"input": graph.get_tensor_by_name(input_name)},
                outputs={"output": graph.get_tensor_by_name(output_name)},
            )

    print(f"SavedModel exported to: {output_dir}")

if __name__ == "__main__":
    main()

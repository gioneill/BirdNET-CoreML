#!/usr/bin/env python3
"""
convert.py – BirdNET Keras→Core ML via a single-signature SavedModel
• coremltools 8.3
• TensorFlow-macOS 2.19.0 (tf.keras 3.10.0)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras as k
import coremltools as ct
from pathlib import Path
import custom_layers

# ──────────────────────────────────────────────────────────────────────────────
#  Shim for old Activation config                                             #
# ──────────────────────────────────────────────────────────────────────────────
class FixedSigmoid(tf.keras.layers.Activation):
    """Fallback for legacy Activation layers saved without an 'activation' key."""
    def __init__(self, activation="sigmoid", **kwargs):
        super().__init__(activation, **kwargs)

CUSTOM_OBJECTS = {
    "SimpleSpecLayer": custom_layers.SimpleSpecLayer,
    "Sigmoid":          FixedSigmoid,
    "Activation":       FixedSigmoid,
}

# ──────────────────────────────────────────────────────────────────────────────
#  Paths                                                                      #
# ──────────────────────────────────────────────────────────────────────────────
KERAS_FILE    = Path("model/BirdNET_6000_RAW_model.keras")
SAVEDMODEL_DIR = Path("model/BirdNET_6000_RAW_savedmodel")
COREML_FILE   = Path("model/BirdNET_6000_RAW.mlpackage")

# ──────────────────────────────────────────────────────────────────────────────
#  Conversion                                                                 #
# ──────────────────────────────────────────────────────────────────────────────
def keras2coreml():
    if not KERAS_FILE.exists():
        raise FileNotFoundError(f"{KERAS_FILE} not found – run your build step first.")

    print("1/3: Loading Keras model…")
    model = k.models.load_model(
        KERAS_FILE,
        compile=False,
        custom_objects=CUSTOM_OBJECTS,
    )

    # Build a single-signature tf.function
    print("2/3: Exporting a single-signature SavedModel…")
    input_name = model.inputs[0].name.split(":")[0]  # typically "INPUT"
    spec = tf.TensorSpec(shape=(1, 144000), dtype=tf.float32, name=input_name)

    @tf.function(input_signature=[spec])
    def serve(x):
        return model(x, training=False)

    # Overwrite any old directory
    if SAVEDMODEL_DIR.exists():
        import shutil
        shutil.rmtree(SAVEDMODEL_DIR)

    tf.saved_model.save(
        model,
        SAVEDMODEL_DIR,
        signatures={"serving_default": serve.get_concrete_function()}
    )

    # Discover the real placeholder
    imported     = tf.saved_model.load(str(SAVEDMODEL_DIR))
    signature    = imported.signatures["serving_default"]
    real_input_name = signature.inputs[0].name.split(":")[0]
    print("3/3: Converting to Core ML…")
    print("   using input placeholder:", real_input_name)

    audio_input = ct.TensorType(name=real_input_name, shape=(1, 144000))

    # ──────────────────────────────────────────────────────────────────────────────
    #  Load class labels and configure classifier
    # ──────────────────────────────────────────────────────────────────────────────
    labels_path = Path("../BirdNET-Analyzer/birdnet_analyzer/labels/V2.4/BirdNET_GLOBAL_6K_V2.4_Labels_en_uk.txt")
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]

    config = ct.ClassifierConfig(
        class_labels=labels,
        predicted_feature_name="classLabel"
        # predicted_probabilities_output="classLabelProbs"
    )

    mlmodel = ct.convert(
        str(SAVEDMODEL_DIR),
        source="tensorflow",
        convert_to="mlprogram",           
        inputs=[audio_input],
        classifier_config=config,             
        minimum_deployment_target=ct.target.iOS15,
        compute_units=ct.ComputeUnit.ALL,
    )

    mlmodel.author            = "Stefan Kahl"
    mlmodel.short_description = "Bird sound recognition with BirdNET."
    mlmodel.save(COREML_FILE)
    print("✓ Saved Core ML model to", COREML_FILE)

if __name__ == "__main__":
    keras2coreml()
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
COREML_FILE   = Path("model/BirdNET_6000_RAW_with_logits.mlpackage") # Changed output filename

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
    print("\n--- Original Keras Model Summary ---")
    model.summary()
    # For debugging, you could print model.summary() here to verify layer names/order
    # model.summary()

    # --- MODIFICATION START: Define a multi-output model ---
    print("INFO: Creating multi-output model to include pre-sigmoid logits.")
    
    # Assumption: The last layer of 'model' is the sigmoid activation.
    # The 'pre_sigmoid_logits' are the outputs of the layer feeding into this sigmoid.
    if not isinstance(model.layers[-1], (tf.keras.layers.Activation, FixedSigmoid)) or \
       (hasattr(model.layers[-1], 'activation') and model.layers[-1].activation.__name__ != 'sigmoid'):
        print("WARNING: The last layer of the loaded Keras model does not appear to be a standalone sigmoid activation.")
        print("         The 'pre_sigmoid_logits' output might not be correctly identified.")
        print(f"         Last layer type: {type(model.layers[-1])}")

    pre_sigmoid_logits_tensor = model.layers[-2].output  # Output of the layer before the final sigmoid (GLOBAL_AVG_POOL)
    original_output_tensor = model.output               # Original model output (after sigmoid layer)

    multi_output_model = k.Model(
        inputs=model.inputs,
        outputs={
            "sigmoid_output": original_output_tensor,       # Name for the Keras sigmoid output
            "pre_sigmoid_logits": pre_sigmoid_logits_tensor # Name for the Keras pre-sigmoid logits output
        },
        name="BirdNET_MultiOutput" # Optional: give the new model a name
    )
    print("INFO: Multi-output model created with outputs: 'sigmoid_output' and 'pre_sigmoid_logits'.")
    print("\n--- Multi-Output Keras Model Summary ---")
    multi_output_model.summary()
    # multi_output_model.summary() # For debugging
    # --- MODIFICATION END ---

    # Build a single-signature tf.function using the multi_output_model
    print("2/3: Exporting a single-signature SavedModel…")
    input_name = multi_output_model.inputs[0].name.split(":")[0] 
    spec = tf.TensorSpec(shape=(1, 144000), dtype=tf.float32, name=input_name)

    @tf.function(input_signature=[spec])
    def serve(x):
        return multi_output_model(x, training=False)

    if SAVEDMODEL_DIR.exists():
        import shutil
        shutil.rmtree(SAVEDMODEL_DIR)

    tf.saved_model.save(
        multi_output_model, 
        SAVEDMODEL_DIR,
        signatures={"serving_default": serve.get_concrete_function()}
    )

    imported     = tf.saved_model.load(str(SAVEDMODEL_DIR))
    signature    = imported.signatures["serving_default"]
    real_input_name = signature.inputs[0].name.split(":")[0]
    print("3/3: Converting to Core ML…")
    print("   using input placeholder:", real_input_name)

    audio_input = ct.TensorType(name=real_input_name, shape=(1, 144000))

    labels_path = Path("../BirdNET-Analyzer/birdnet_analyzer/labels/V2.4/BirdNET_GLOBAL_6K_V2.4_Labels_en_uk.txt")
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]

    # Configure ClassifierConfig to guide Core ML
    config = ct.ClassifierConfig(
        labels,  # This is class_labels
        predicted_feature_name="sigmoid_output"  # Tell Core ML to use the Keras output named "sigmoid_output" for classification
        # predicted_probabilities_output="classLabelProbs" # Removed this line
    )

    mlmodel = ct.convert(
        str(SAVEDMODEL_DIR),
        source="tensorflow",
        convert_to="mlprogram",           
        inputs=[audio_input],
        classifier_config=config, # Pass the explicit classifier configuration
        minimum_deployment_target=ct.target.iOS15,
        compute_units=ct.ComputeUnit.ALL,
    )

    mlmodel.author            = "Stefan Kahl"
    mlmodel.short_description = "Bird sound recognition with BirdNET (includes pre-sigmoid logits, explicit classifier v2)." # Updated description
    mlmodel.save(COREML_FILE) 
    print("✓ Saved Core ML model to", COREML_FILE)

if __name__ == "__main__":
    keras2coreml()

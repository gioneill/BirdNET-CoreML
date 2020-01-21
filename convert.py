import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import json
import numpy as np

import tensorflow as tf
from tensorflow import keras as k

import tfcoreml
import coremltools

import custom_layers

KERAS_MODEL_FILE = 'model/BirdNET_1000_RAW_model.h5'
KERAS_MODEL_CONFIG = 'model/BirdNET_1000_RAW_config.json'

COREML_MODEL_FILE = 'model/BirdNET_1000_RAW.mlmodel'

def loadKerasModel(h5file, config_file, layer_index=-1):
    
    print('LOADING MODEL FROM CHECKPOINT', h5file.split(os.sep)[-1], '...')

    # Load trained net
    net = k.models.load_model(h5file,
                              custom_objects={'SimpleSpecLayer': custom_layers.SimpleSpecLayer},
                              compile=False)

    # Select specific output layer
    if not layer_index == -1:
        net = k.Model(net.inputs, net.layers[layer_index].output)

    # Get class labels from config
    with open(config_file, 'r') as cfile:
        config = json.load(cfile)
        labels = config['CLASSES']

    return net, labels

def kerasToCoreMLSimpleSpecLayer(kerasLayer):                                                                       
    coreml_layer = coremltools.proto.NeuralNetwork_pb2.CustomLayerParams()
    coreml_layer.className = 'SimpleSpecLayer'
    return coreml_layer

def keras2coreml(keras_model_path, cml_model_path, config_path, layer_index=-1):

    # Load keras model
    keras_model, labels = loadKerasModel(keras_model_path, config_path, layer_index)

    # Export as protobuf model (it seems like CoreML requires this in order to also convert custom layers)
    print('SAVING AS TF MODEL DIR...')
    k.models.save_model(
        keras_model,
        keras_model_path.rsplit('.', 1)[0],
        overwrite=True,
        include_optimizer=False,
        save_format='tf',
        signatures=None,
        options=None
    )

    # Get input, output node names for the TF graph from the Keras model
    print('CONVERTING TO COREML...')
    input_name = keras_model.inputs[0].name.split(':')[0]
    keras_output_node_name = keras_model.outputs[0].name.split(':')[0]
    graph_output_node_name = keras_output_node_name.split('/')[-1]
    print(input_name, graph_output_node_name)

    # Convert this model to Core ML format
    cml_model = coremltools.converters.tensorflow.convert(
                            #keras_model,
                            filename=keras_model_path.rsplit('.', 1)[0] + '/saved_model.pb',
                            #tf_model_path=keras_model,
                            input_name_shape_dict={input_name: (144000)}, # Sample rate * signal length = 48000 * 3 = 144000
                            output_feature_names=[graph_output_node_name],
                            minimum_ios_deployment_target='13',
                            add_custom_layers=True, 
                            custom_conversion_functions={'SimpleSpecLayer': kerasToCoreMLSimpleSpecLayer}
                            )

    cml_model.author = 'Stefan Kahl'
    cml_model.short_description = 'Bird sound recognition with BirdNET.'

    # We could probably use the labels
            
    cml_model.save(cml_model_path)

if __name__ == '__main__':

    keras2coreml(KERAS_MODEL_FILE, COREML_MODEL_FILE, KERAS_MODEL_CONFIG)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras import layers as l

import custom_layers

RANDOM_SEED = 1337

# Model config
FILTERS = [8, 16, 32, 64, 128]
KERNEL_SIZES =  [(5, 5), (3, 3), (3, 3), (3, 3), (3, 3)]
BRANCH_KERNEL_SIZE = (4, 10)
RESNET_K = 2
RESNET_N = 3
ACTIVATION = 'relu'
INITIALIZER = 'he'
DROPOUT_RATE = 0.33
NUM_CLASSES = 1000

# Input config
SAMPLE_RATE = 48000
SPEC_LENGTH = 3.0 # 3 seconds
SPEC_SHAPE = (257, 384) # (height, width)

# Initializers
initializers = {'glorot': tf.initializers.GlorotNormal(seed=RANDOM_SEED),
                'he': k.initializers.he_normal(seed=RANDOM_SEED),
                'random': k.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=RANDOM_SEED),
                'constant': k.initializers.Constant(value=0)
                }

# Activations
activations = {'relu': l.ReLU(max_value=None, negative_slope=0.0, threshold=0.0),
               'elu': l.ELU(alpha=1.0),
               'lrelu': l.LeakyReLU(alpha=0.3)
              }

def resBlock(net_in, filters, kernel_size, stride=1, preactivated=True, block_id=1, name=''):

    # Show input shape
    print('    ' + name + ' IN SHAPE:', net_in.shape, end=' ')

    # Pre-activation
    if block_id > 1:
        net_pre = l.BatchNormalization(axis=-1, name=name + '_BN_PA')(net_in)
        net_pre = activations[ACTIVATION](net_pre)
    else:
        net_pre = net_in    

    # Pre-activated shortcut?
    if preactivated:
        net_in = net_pre

    # Bottleneck Convolution
    if stride > 1:
        net_pre = l.Conv2D(filters=net_pre.shape[-1],
                            kernel_size=1,
                            padding='same',
                            strides=1,
                            data_format='channels_last',
                            kernel_initializer=initializers[INITIALIZER],
                            name=name + '_CONV_1')(net_pre)
        net_pre = l.BatchNormalization(axis=-1, name=name + '_BN_1')(net_pre)
        net_pre = activations[ACTIVATION](net_pre)
    
    # First Convolution     
    net = l.Conv2D(filters=net_pre.shape[-1],
                   kernel_size=kernel_size,
                   padding='same',
                   strides=1,
                   data_format='channels_last',
                   kernel_initializer=initializers[INITIALIZER],
                   name=name + '_CONV_2')(net_pre)
    net = l.BatchNormalization(axis=-1, name=name + '_BN_2')(net)
    net = activations[ACTIVATION](net)

    # Pooling layer
    if stride > 1:
        net = l.MaxPooling2D(pool_size=(stride, stride), 
                             data_format='channels_last', 
                             name=name + '_POOL_1')(net)

    # Dropout Layer
    net = l.Dropout(rate=DROPOUT_RATE, seed=RANDOM_SEED, name=name + '_DO_1')(net)        

    # Second Convolution (make 1x1 if downsample block)
    if stride > 1:
        k_size = (1, 1)
    else:
        k_size =  kernel_size
    net = l.Conv2D(filters=filters,
                   kernel_size=k_size,
                   padding='same',
                   strides=1,
                   activation=None,
                   data_format='channels_last',
                   kernel_initializer=initializers[INITIALIZER],
                   name=name + '_CONV_3')(net)
    
    # Shortcut Layer
    if not net.shape[1:] == net_in.shape[1:]:        

        # Average pooling
        shortcut = l.AveragePooling2D(pool_size=(stride, stride), 
                                      data_format='channels_last',  
                                      name=name + '_SC_POOL')(net_in)

        # Shortcut convolution
        shortcut = l.Conv2D(filters=filters,
                            kernel_size=1,
                            padding='same',
                            strides=1,
                            activation=None,
                            data_format='channels_last',
                            kernel_initializer=initializers[INITIALIZER],
                            name=name + '_SC_CONV')(shortcut)
        
    else:

        # Shortcut = input
        shortcut = net_in
    
    # Merge Layer
    out = l.add([net, shortcut], name=name + '_ADD')

    # Show output shape
    print('OUT SHAPE:', out.shape)

    return out

def classificationBranch(net, kernel_size):

    # Post Convolution
    branch = l.Conv2D(filters=int(FILTERS[-1] * RESNET_K),
                      kernel_size=kernel_size,
                      strides=1,
                      data_format='channels_last',
                      kernel_initializer=initializers[INITIALIZER],
                      name='BRANCH_CONV_1')(net)
    branch = l.BatchNormalization(axis=-1, name='BRANCH_BN_1')(branch)
    branch = activations[ACTIVATION](branch)

    print('    POST  CONV SHAPE:', branch.shape)

    # Dropout Layer
    branch = l.Dropout(rate=DROPOUT_RATE, seed=RANDOM_SEED, name='BRANCH_DO_1')(branch)   
    
    # Dense Convolution
    branch = l.Conv2D(filters=int(FILTERS[-1] * RESNET_K * 2),
                      kernel_size=1,
                      strides=1,
                      data_format='channels_last',
                      kernel_initializer=initializers[INITIALIZER],
                      name='BRANCH_CONV_2')(branch)
    branch = l.BatchNormalization(axis=-1, name='BRANCH_BN_2')(branch)
    branch = activations[ACTIVATION](branch)

    print('    DENSE CONV SHAPE:', branch.shape)
    
    # Dropout Layer
    branch = l.Dropout(rate=DROPOUT_RATE, seed=RANDOM_SEED, name='BRANCH_DO_2')(branch)     

    # Class Convolution
    branch = l.Conv2D(filters=NUM_CLASSES,
                      kernel_size=1,
                      activation=None,
                      data_format='channels_last',
                      kernel_initializer=initializers[INITIALIZER],
                      name='BRANCH_CONV_3_' + str(NUM_CLASSES))(branch)

    return branch

def buildModel():

    print('BUILDING BirdNET MODEL...')

    # Input layer
    print('  INPUT:')
    inputs = k.Input(shape=int(SAMPLE_RATE * SPEC_LENGTH), name='INPUT')

    # Spectrogram layer if input is raw signal
    net = custom_layers.SimpleSpecLayer(sample_rate=SAMPLE_RATE,
                                spec_shape=SPEC_SHAPE,
                                frame_step=int(SAMPLE_RATE * SPEC_LENGTH / (SPEC_SHAPE[1] + 1)),
                                data_format='channels_last',
                                name='SIMPLESPEC')(inputs)

    print('    INPUT LAYER  IN SHAPE:', inputs.shape)                       
    print('    INPUT LAYER OUT SHAPE:', net.shape)

    # Preprocessing convolution
    print('  PRE-PROCESSING STEM:')
    net = l.Conv2D(filters=int(FILTERS[0] * RESNET_K),
                   kernel_size=KERNEL_SIZES[0],
                   strides=(2, 1),
                   padding='same',
                   data_format='channels_last',
                   kernel_initializer=initializers[INITIALIZER],
                   name='CONV_0')(net)

    # Batch norm layer
    net = l.BatchNormalization(axis=-1, name='BNORM_0')(net)
    net = activations[ACTIVATION](net)
    print('    FIRST  CONV OUT SHAPE:', net.shape)

    # Max pooling
    net = l.MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name='POOL_0')(net)
    print('    FIRST  POOL OUT SHAPE:', net.shape)

    # Residual Stacks
    for i in range(1, len(FILTERS)):
        print('  RES STACK', i, ':')
        net = resBlock(net,
                       filters=int(FILTERS[i] * RESNET_K),
                       kernel_size=KERNEL_SIZES[i],
                       stride=2,
                       preactivated=True,
                       block_id=i,
                       name='BLOCK_' + str(i) + '-1')
        
        for j in range(1, RESNET_N):
            net = resBlock(net,
                           filters=int(FILTERS[i] * RESNET_K),
                           kernel_size=KERNEL_SIZES[i],
                           preactivated=False,
                           block_id=i+j,
                           name='BLOCK_' + str(i) + '-' + str(j + 1))

    # Post Activation
    net = l.BatchNormalization(axis=-1, name='BNORM_POST')(net)  
    net = activations[ACTIVATION](net)  

    # Classification branch
    print('  CLASS BRANCH:')
    net = classificationBranch(net,  BRANCH_KERNEL_SIZE) 
    print('    BRANCH OUT SHAPE:', net.shape)

    # Pooling
    net = l.GlobalAveragePooling2D(data_format='channels_last', name='GLOBAL_AVG_POOL')(net)
    print('  GLOBAL POOLING SHAPE:', net.shape)

    # Classification layer
    outputs = k.activations.sigmoid(net)
    print('  FINAL NET OUT SHAPE:', outputs.shape)

    # Build Keras model
    model = k.Model(inputs=inputs, outputs=outputs, name='BirdNET')

    # Print model stats
    print('...DONE!')
    #log.l(model.summary()) 
    
    print('MODEL HAS', (len([layer for layer in model.layers if len(layer.get_weights()) > 0])), 'WEIGHTED LAYERS (', len(model.layers), 'TOTAL )')
    print('MODEL HAS', model.count_params(), 'PARAMS')

    return model

def saveModel(model, name):

    print('SAVING MODEL...', end='')
    model.save(os.path.join('model', name)) 
    print('DONE!')

if __name__ == '__main__':

    model = buildModel()
    saveModel(model, 'BirdNET_1000_RAW_model.h5')


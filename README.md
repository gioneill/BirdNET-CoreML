# BirdNET-CoreML
CoreML conversion of trained BirdNET models

## Install requirements (Ubuntu 18.04)

TF 2.1 needs python > 3.4 and pip > 19.0

```
sudo apt-get update
sudo apt-get install python3-dev python3-pip
sudo pip3 install --upgrade pip
sudo pip3 install tensorflow
```

CoreML Tools

```
sudo pip3 install --upgrade tfcoreml coremltools
```

## Build model

Run ```build_model.py``` to build and save a Keras model for BirdNET. At the top of the file, some config options can be adjusted:

```
FILTERS = [8, 16, 32, 64, 128]
KERNEL_SIZES =  [(5, 5), (3, 3), (3, 3), (3, 3), (3, 3)]
BRANCH_KERNEL_SIZE = (4, 10)
RESNET_K = 2
RESNET_N = 3
ACTIVATION = 'relu'
INITIALIZER = 'he'
DROPOUT_RATE = 0.33
NUM_CLASSES = 1000
```

# 2020-10-09 22:29:26
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
import numpy as np
seed = 1
np.random.seed(seed)

def vgg16(weights_path=None):
    #input (224 × 224 RGB image)
    input_shape = (224, 224, 3)
    model = Sequential()
    #conv3-64
    model.add(Conv2D(64, (3,3), strides = (1,1), input_shape = input_shape, padding = 'same', activation = 'relu', kernel_initializer = 'uniform'))
    model.add(Conv2D(64, (3,3), strides = (1,1), padding = 'same', activation = 'relu', kernel_initializer = 'uniform'))
    #maxpool 
    model.add(MaxPooling2D(pool_size = (2,2), strides = None))
    #conv3-128
    model.add(Conv2D(128, (3,3), strides = (1,1), padding = 'same', activation = 'relu', kernel_initializer = 'uniform'))
    model.add(Conv2D(128, (3,3), strides = (1,1), padding = 'same', activation = 'relu', kernel_initializer = 'uniform'))
    #maxpool
    model.add(MaxPooling2D(pool_size = (2,2), strides = None))
    #conv3-256
    model.add(Conv2D(256, (3,3), strides = (1,1), padding = 'same', activation = 'relu', kernel_initializer = 'uniform'))
    model.add(Conv2D(256, (3,3), strides = (1,1), padding = 'same', activation = 'relu', kernel_initializer = 'uniform'))
    model.add(Conv2D(256, (3,3), strides = (1,1), padding = 'same', activation = 'relu', kernel_initializer = 'uniform'))
    #maxpool
    model.add(MaxPooling2D(pool_size = (2,2), strides = None))
    ##conv3-512
    model.add(Conv2D(512, (3,3), strides = (1,1), padding = 'same', activation = 'relu', kernel_initializer = 'uniform'))
    model.add(Conv2D(512, (3,3), strides = (1,1), padding = 'same', activation = 'relu', kernel_initializer = 'uniform'))
    model.add(Conv2D(512, (3,3), strides = (1,1), padding = 'same', activation = 'relu', kernel_initializer = 'uniform'))
    #maxpool
    model.add(MaxPooling2D(pool_size = (2,2), strides = None))
    ##conv3-512
    model.add(Conv2D(512, (3,3), strides = (1,1), padding = 'same', activation = 'relu', kernel_initializer = 'uniform'))
    model.add(Conv2D(512, (3,3), strides = (1,1), padding = 'same', activation = 'relu', kernel_initializer = 'uniform'))
    model.add(Conv2D(512, (3,3), strides = (1,1), padding = 'same', activation = 'relu', kernel_initializer = 'uniform'))
    #maxpool
    model.add(MaxPooling2D(pool_size = (2,2), strides = None))
    #
    model.add(Flatten())
    #FC-4096
    model.add(Dense(4096, activation = 'relu'))
    #dropout 
    model.add(Dropout(0.5))
    #FC-4096
    model.add(Dense(4096, activation = 'relu'))
    #dropout
    model.add(Dropout(0.5))
    #FC-1000
    model.add(Dense(1000, activation = 'relu'))
    if weights_path:
        model.load(weights_path)
    return model

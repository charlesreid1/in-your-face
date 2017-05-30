from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils

# Convolutional neural network architecture:
#     FC/Dense

def phoenix_model(  batch_size = 128,
                    nb_classes = 2,
                    nb_epoch = 1,
                    feature_maps = 32,
                    feature_maps_size = (3,3),
                    input_width = 32,
                    input_height = 32):

    assert(len(feature_maps_size)==2)

    # 2 images x 3 channels = 6 input channels 
    input_shape = (6, input_width, input_height)

    w = Sequential()

    phoenix.add( Flatten() )

    phoenix.add( Dense(2, activation='softmax', name='predictions') )
    
    return phoenix

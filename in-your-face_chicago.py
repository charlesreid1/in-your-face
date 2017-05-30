from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils

# Convolutional neural network architecture:
#     Convolution, Activation
#     Pooling
#     Convolution, Activation
#     Pooling
#     FC/Dense
#     Activation
#     FC/Dense
#     Activation

def chicago_model(  batch_size = 128,
                    nb_classes = 2,
                    nb_epoch = 1,
                    feature_maps = 32,
                    feature_maps_size = (3,3),
                    input_width = 32,
                    input_height = 32):

    assert(len(feature_maps_size)==2)

    # 2 images x 3 channels = 6 input channels 
    input_shape = (6, input_width, input_height)

    chicago = Sequential()

    # Convolutional input layer,
    # specify number of feature maps,
    # size of feature maps,
    # image sizes are (6, 32, 32) - 6 color channels, 32 x 32 pixels

    #1 Convulution, Activation, Pooling
    chicago.add( Conv2D(feature_maps, 
                        feature_maps_size, 
                        input_shape = input_shape,
                        data_format = "channels_first",
                        padding = 'valid') )

    chicago.add( Activation('relu') )

    chicago.add( MaxPooling2D(data_format="channels_first", pool_size=(2, 2)) )


    #2 Convulution, Activation, Pooling
    chicago.add( Conv2D(feature_maps, 
                        feature_maps_size, 
                        input_shape = input_shape,
                        data_format = "channels_first",
                        padding = 'valid') )

    chicago.add( Activation('relu') )

    chicago.add( MaxPooling2D(data_format="channels_first", pool_size=(2, 2)) )


    #1 FC/Dense, Activation
    chicago.add( Flatten() )

    chicago.add( Dense(128, activation='relu') )
    
    chicago.add( Activation('relu') )


    #2 FC/Dense, Activation
    chicago.add( Flatten() ) 

    chicago.add( Dense(2, activation='softmax', name='predictions') )

    chicago.add( Activation('relu') )

    return chicago

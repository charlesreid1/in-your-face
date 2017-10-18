from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils

# Convolutional neural network architecture:
#     Convolution
#     (No Dropout)
#     Convolution
#     Pool
#     Dropout
#     Flatten
#     Dense
#     Dropout
#     Dense

def seattle_model(  batch_size = 128,
                    nb_classes = 2,
                    nb_epoch = 1,
                    feature_maps = 32,
                    feature_maps_size = (3,3),
                    input_width = 32,
                    input_height = 32):

    assert(len(feature_maps_size)==2)

    # 2 images x 3 channels = 6 input channels 
    input_shape = (6, input_width, input_height)

    seattle = Sequential()

    # Convolutional input layer:
    # - Specify number of feature maps
    # - Specify size of feature maps
    # - Image sizes are known: (6, 32, 32)
    # - That's 6 color channels, 32 x 32 pixels
    # 
    # - What is this "channels_first" option?
    # 
    seattle.add( Conv2D(feature_maps, 
                        feature_maps_size, 
                        input_shape = input_shape,
                        data_format = "channels_first",
                        padding = 'valid') )
    
    ## Not clear if batch normalization needed...
    #seattle.add( BatchNormalization(axis=1, scale=False) )
    seattle.add( Activation('relu') )

    # Second convolutional layer
    seattle.add( Conv2D(feature_maps, 
                        feature_maps_size, 
                        input_shape = input_shape,
                        data_format = "channels_first",
                        padding = 'valid') )
    
    #seattle.add( BatchNormalization(axis=1, scale=False) )
    seattle.add( Activation('relu') )

    # Max Pool layer with size 2 × 2.
    seattle.add( MaxPooling2D(data_format="channels_first",
                              pool_size=(2, 2)) )
    
    ## Alternative: use an Avg Pool layer
    #seattle.add( AveragePooling2D(pool_size=(2, 2)) )

    # Dropout set to 20%
    seattle.add( Dropout(0.2) )

    # Flatten the space-specific nodes
    # (What's the difference between Flatten and Dense?)
    seattle.add( Flatten() )

    # Fully connected layer with 128 units 
    seattle.add( Dense(128, activation='relu') )

    # Dropout set to 50%
    seattle.add( Dropout(0.50) )

    # Fully connected output layer,
    # 1 unit (yes/no)
    seattle.add( Dense(1, activation='softmax', name='predictions') )

    return seattle


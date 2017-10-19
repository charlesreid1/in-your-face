from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.constraints import maxnorm

"""





"""

def seattle_model(  input_width = 32,
                    input_height = 32,
                    feature_maps = 32,
                    feature_window_size = (5,5),
                    dropout1 = 0.2,
                    dense = 128,
                    dropout2 = 0.5,
                    use_max_pooling = True,
                    pool_size = (2,2),
                    optimizer = 'rmsprop' ):
    """
    Seattle Model Architecture:
    
        Convolution
        Convolution
        Pool
        Dropout
        Flatten
        Dense
        Dropout
        Dense

    Input arguments:

        input_width/input_height are size of image, obviously

        feature_maps/feature_window_size determine the number and size of convolution

        dropout1 is number of nodes after pooling layer to set to 0

        dropout2 is number of nodes after flatten/dense layer to set to 0

        dense is number of nodes in dense layer

        use_max_pooling is a boolean; if yes, use MaxPooling, else use AveragePooling

        optimizer is one of 'rmsprop'
    """

    modelA = Sequential()
    
    # Convolutional input layer:
    # - 20 feature maps (each feature map is a reduced-size convolution that detects a different feature)
    # - 3 pixel square window
    modelA.add(Conv2D(feature_maps, 
                      feature_window_size,
                      input_shape=(ds,ds,6),
                      padding='same',
                      data_format='channels_last',
                      activation='relu'))
    
    # Second convolutional layer
    # - 40 feature maps (add more features)
    # - 3 pixel square window
    modelA.add(Conv2D(feature_maps, 
                      feature_window_size,
                      padding='same',
                      data_format='channels_last',
                      activation='relu'))
    
    # Pooling layer
    if(use_max_pooling):
        modelA.add(MaxPooling2D(pool_size=pool_size,
                                data_format='channels_last'))
    else:
        modelA.add(AveragePooling2D(pool_size=pool_size,
                                    data_format='channels_last'))
    
    # Set X% of units to 0
    modelA.add(Dropout(0.2))
    
    # Flatten layer
    modelA.add(Flatten())
    
    # Fully connected layer with 128 units and a rectifier activation function.
    modelA.add( Dense(128, 
                activation='relu', 
                kernel_constraint=maxnorm(3)))
    
    # Dropout set to 50%.
    modelA.add(Dropout(0.5))
    
    # Fully connected output layer with 2 units (Y/N)
    # and a softmax activation function.
    modelA.add(Dense(1, activation='sigmoid'))
    
    if(optimizer not in ['rmsprop','adam','adadelta']):
        optimizer = 'rmsprop'

    model.compile(  loss='binary_crossentropy', 
                    metrics=['binary_accuracy'], 
                    optimizer='rmsprop')
                   



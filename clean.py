import numpy as np
from scipy.misc import imresize



def clean( X_train_original, y_train_original,
           X_test_original,  y_test_original,
           downsample = 32):
    """
    Clean the original data loaded by lfw_fuel.
    """

    X_train = np.asarray([crop_and_downsample(x, downsample_size=downsample) for x in X_train_original])
    X_test  = np.asarray([crop_and_downsample(x, downsample_size=downsample) for x in X_test_original])
    y_train = y_train_original
    y_test  = y_test_original

    return (X_train,y_train), (X_test,y_test)



def crop_and_downsample(originalX, downsample_size=32):
    """
    Starts with a 250 x 250 image.
    Crops to 128 x 128 around the center.
    Downsamples the image to (downsample_size) x (downsample_size).
    Returns an image with dimensions (channel, width, height).
    """
    current_dim = 250
    target_dim = 128
    margin = int((current_dim - target_dim)/2)
    left_margin = margin
    right_margin = current_dim - margin

    # newim is shape (6, 128, 128)
    newim = originalX[:, left_margin:right_margin, left_margin:right_margin]

    # resized are shape (feature_width, feature_height, 3)
    feature_width = feature_height = downsample_size
    resized1 = imresize(newim[0:3,:,:], (feature_width, feature_height), interp="bicubic", mode="RGB")
    resized2 = imresize(newim[3:6,:,:], (feature_width, feature_height), interp="bicubic", mode="RGB")

    # re-packge into a new X entry
    newX = np.concatenate([resized1,resized2], axis=2)

    # the next line is EXTREMELY important.
    # if you don't normalize your data, all predictions will be 0 forever.
    newX = newX/255

    return newX


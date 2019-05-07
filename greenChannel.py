from importImage import *
from skimage import filters
import numpy as np


def eliminateZero(matrix):
    """ Takes in a 2D array representing the green channel
    and removes the zero values

    :param matrix: 2D array representing the green channel.
    :type: 2D array
    :return: 2D array with nonzero pixel values.
    """

    new_a = matrix[np.nonzero(matrix)]
    return new_a


def otsu(matrix):
    """Takes in a 2D array (ideally the green channel) and
    finds the otsu threshold pixel value of the array.

    :param matrix: 2D array ideally representing the green channel.
    :type: 2D array.
    :return: Single value representing the otsu threshold.
    """

    val = filters.threshold_otsu(matrix)
    return val


if __name__ == "__main__":
    (red, green, blue) = convertRGB("ExampleAbnormalCervix.tif")
    nonzero_red = eliminateZero(red)
    res = otsu(nonzero_red)
    print(res)

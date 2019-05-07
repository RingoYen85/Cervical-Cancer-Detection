from PIL import Image
import numpy as np
from scipy import stats


def convertRGB(filename):
    """Take TIF filename as input, split image into RGB
    contents and convert them into arrays

    :param filename: filename for image
    :type filename: String
    :returns: tuple of red, green and blue arrays
    """

    im = Image.open(filename)
    rgb = im.split()
    red = np.array(rgb[0])
    green = np.array(rgb[1])
    blue = np.array(rgb[2])

    return red, green, blue


def remove_specular(blue):
    """ Takes in a numpy array (a 2D array) and removes the pixel
    values that are greater than 240, by setting them to zero.

    :param blue:
    :type blue: Numpy 2D array;
    :returns: A numpy 2D array representing the blue channel,
    but without the 240 values.
    """

    blue = np.array(blue)
    row, col = blue.shape

    for x in range(0, row):
        for c in range(0, col):
            if blue[x, c] > 240:
                blue[x, c] = 0

    return blue


def remove_zero(blue):
    """Takes in a 2D array representing the blue channel (ideally with the specular
    reflection removed) and removes all the zero values in the array

    :param blue: 2D array representing the blue channel of the image.
    :type: numpy 2D array.
    :return: numpy array representing all nonzero values.
    """

    no_zero_blue = blue[np.nonzero(blue)]
    return no_zero_blue


def find_mean(no_zero_blue):
    """Takes in a 2D array representing the blue channel and
    finds the mean pixel value.

    :param no_zero_blue: blue channel array
    :type: 2D numpy array
    :return: Mean pixel value of array.
    """

    no_zero_blue = np.array(no_zero_blue).flatten()
    blue_total = np.sum(no_zero_blue)
    blue_length = len(no_zero_blue)

    blue_mean = blue_total / blue_length

    return blue_mean


def find_mode(no_zero_blue):
    """Takes in a 2D array representing the blue channel
    and finds the mode of the pixel values.

    :param no_zero_blue:
    :type: 2D numpy array
    :return: Mode of the pixel values.
    """

    no_zero_blue = np.array(no_zero_blue).flatten()
    blue_mode, mode_count = stats.mode(no_zero_blue)
    return blue_mode[0]


def find_median(no_zero_blue):
    """Takes in a 2D array representing the blue channel
    and finds the median pixel value.
    The array is flattened the numpy median function is applied.

    :param no_zero_blue: 2D numpy array
    :type: 2D numpy array
    :return: Median pixel value.
    """
    no_zero_blue = no_zero_blue.flatten()
    blue_median = np.median(no_zero_blue)

    return blue_median


if __name__ == "__main__":
    r, g, b = convertRGB("ExampleAbnormalCervix.tif")
    b_nospec = remove_specular(b)

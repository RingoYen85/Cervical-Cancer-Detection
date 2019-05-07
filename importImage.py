from PIL import Image
import numpy as np


def convertRGB(filename):
    """Take TIF filename as input, split image into RGB
    contents and convert them into arrays

    :param filename: filename for image
    :type filename: String
    :returns: tuple of red, green and blue arrays
    """

    im = Image.open(filename)
    # im.show()
    print(type(im))
    rgb = im.split()
    red = np.array(rgb[0])
    green = np.array(rgb[1])
    blue = np.array(rgb[2])
    print(blue)

    return red, green, blue


if __name__ == "__main__":
    r, g, b = convertRGB("unitTest_2.tif")

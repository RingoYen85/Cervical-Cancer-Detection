import numpy as np
from scipy.misc import imsave

# python file just to make 2 test images for unit testing.
data_1= np.array([[[0,255,0], [0,0,0], [255,255,255]],
                 [[0, 255, 255], [0, 0, 255], [10, 255, 255]]])

data_2 = np.array([[[0,0,255], [0,255,0], [255,0,0]],
                 [[0,0,255], [0, 255,0], [255,0,0]]])

imsave('unitTest_1.tif', data_1)
imsave('unitTest_2.tif',data_2)




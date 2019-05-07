import numpy as np
import pytest
from blueChannelProcessing import *
from svm import *;


# working with 2D arrays.
class TestClass:

    def test_convertRGB(self):

        actual_red_1 = np.array([[0, 0, 255], [0, 0, 10]])
        actual_green_1 = np.array([[255, 0, 255], [255, 0, 255]])
        actual_blue_1 = np.array([[0, 0, 255], [255, 255, 255]])

        actual_red_2 = np.array([[0, 0, 255], [0, 0, 255]])
        actual_green_2 = np.array([[0, 255, 0], [0, 255, 0]])
        actual_blue_2 = np.array([[255, 0, 0], [255, 0, 0]])

        r1, g1, b1 = convertRGB("unitTest_1.tif")
        r2, g2, b2 = convertRGB("unitTest_2.tif")

        np.testing.assert_equal(actual_red_1, r1)
        np.testing.assert_equal(actual_green_1, g1)
        np.testing.assert_equal(actual_blue_1, b1)

        np.testing.assert_equal(actual_red_2, r2)
        np.testing.assert_equal(actual_green_2, g2)
        np.testing.assert_equal(actual_blue_2, b2)

    def test_remove_specular(self):

        # input will be a 2D np array. no need to use np matrix.
        test_matrix_1 = np.array([[70, 0, 50, 240], [65, 1, 241, 240], [45, 2, 10, 246]])
        test_matrix_2 = np.array([[0, 0, 0 ,241], [1,1,1,1], [242, 242, 242, 245]])
        test_matrix_3 = np.array([[1, 1, 1], [1,1,1], [1,1,1]])
        test_matrix_4 = np.array([[241,241,241], [241, 241, 241], [241, 241, 241]])
        test_matrix_5 = np.array([[1,2,3],[4,5,241]])

        # Possibly try using nan instead of zero, depending on results of SVM.
        # n = float('nan')

        actual_result_1 = np.array([[70, 0, 50, 240], [65,1, 0, 240], [45, 2, 10, 0]])
        actual_result_2 = np.array([[0, 0, 0 ,0], [1,1,1,1], [0, 0, 0, 0]])
        actual_result_3 = np.array([[1, 1, 1], [1,1,1], [1,1,1]])
        actual_result_4 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        actual_result_5 = np.array([[1,2,3], [4,5,0]])


        function_result_1 = remove_specular(test_matrix_1)
        function_result_2 = remove_specular(test_matrix_2)
        function_result_3 = remove_specular(test_matrix_3)
        function_result_4 = remove_specular(test_matrix_4)
        function_result_5 = remove_specular(test_matrix_5)

        np.testing.assert_equal(actual_result_1, function_result_1)
        np.testing.assert_equal(actual_result_2, function_result_2)
        np.testing.assert_equal(actual_result_3, function_result_3)
        np.testing.assert_equal(actual_result_4, function_result_4)
        np.testing.assert_equal(actual_result_5, function_result_5)

    def test_remove_zero(self):

        test_matrix_1 = np.array([[1, 1, 1], [1,1,1], [1,1,1]])
        test_matrix_2 = np.array([[ 0,0,0],[0,0,0], [0,0,0]])
        test_matrix_3 = np.array([[10,20,0,0], [0, 0, 0, 0], [11, 11, 11, 11]])
        test_matrix_4 = np.array([[5, 5, 5, 5], [5,0,0,0], [0, 0, 0, 0]])

        actual_result_1 = np.array([1, 1, 1, 1,1,1, 1,1,1])
        actual_result_2 = np.array([])
        actual_result_3 = np.array([10, 20, 11,11,11, 11])
        actual_result_4 = np.array([5, 5, 5, 5, 5])

        function_result_1 = remove_zero(test_matrix_1)
        function_result_2 = remove_zero(test_matrix_2)
        function_result_3 = remove_zero(test_matrix_3)
        function_result_4 = remove_zero(test_matrix_4)


        np.testing.assert_equal(actual_result_1, function_result_1)
        np.testing.assert_equal(actual_result_2, function_result_2)
        np.testing.assert_equal(actual_result_3, function_result_3)
        np.testing.assert_equal(actual_result_4, function_result_4)


    def test_find_mean(self):

        test_matrix_1 = np.array([0,0, 0, 0, 0])
        test_matrix_2 = np.array([1,1,1,1,1,1,1,1,1,])
        test_matrix_3 = np.array([[5,4,3], [5,0,5], [10,10,10]])
        test_matrix_4 = np.array([2,4,6,6,8,10,2,4,6])

        actual_result_1 = 0.0
        actual_result_2 = 1.0
        actual_result_3 = 52/9
        actual_result_4 = 48/9

        function_result_1 = find_mean(test_matrix_1)
        function_result_2 = find_mean(test_matrix_2)
        function_result_3 = find_mean(test_matrix_3)
        function_result_4 = find_mean(test_matrix_4)

        np.testing.assert_equal(actual_result_1, function_result_1)
        np.testing.assert_equal(actual_result_2, function_result_2)
        np.testing.assert_equal(actual_result_3, function_result_3)
        np.testing.assert_equal(actual_result_4, function_result_4)


    def test_find_mode(self):

        test_array_1 = np.array([1,1,1,1,1])
        test_array_2 = np.array([[2,2,2,3,3],[3,3,3,3,3],[10,10,9,9,9]])
        test_array_3 = np.array([2,2,2,10,10,10])


        actual_result_1 = 1
        actual_result_2 = 3
        actual_result_3 = 2


        function_result_1 = find_mode(test_array_1)
        function_result_2 = find_mode(test_array_2)
        function_result_3 = find_mode(test_array_3)

        np.testing.assert_equal(actual_result_1, function_result_1)
        np.testing.assert_equal(actual_result_2, function_result_2)
        np.testing.assert_equal(actual_result_3, function_result_3)

    def test_find_median(self):

        test_array_1 = np.array([1,1,10,5,6,2])
        test_array_2 = np.array([0,0,0,0,0])
        test_array_3 = np.array([2,2,2,2,2])
        test_array_4 = np.array([[1,1,1,4,4],[5,5,5,5,10], [10,10,10,10,10]])

        actual_result_1 = 3.5
        actual_result_2 = 0
        actual_result_3 = 2
        actual_result_4 = 5

        function_result_1 = find_median(test_array_1)
        function_result_2 = find_median(test_array_2)
        function_result_3 = find_median(test_array_3)
        function_result_4 = find_median(test_array_4)

        np.testing.assert_equal(actual_result_1, function_result_1)
        np.testing.assert_equal(actual_result_2, function_result_2)
        np.testing.assert_equal(actual_result_3, function_result_3)
        np.testing.assert_equal(actual_result_4, function_result_4)

    def test_Diagnosis(self):

        actual_result_1 = "diseased"
        actual_result_2 = "diseased"
        actual_result_3 = "healthy"
        actual_result_4 = "healthy"

        filename_1 = "dysplasia00.tif"
        filename_2 = "dysplasia01.tif"
        filename_3 = "healthy01.tif"
        filename_4 = "healthy02.tif"

        output_1 = diagnosis(filename_1)
        output_2 = diagnosis(filename_2)
        output_3 = diagnosis(filename_3)
        output_4 = diagnosis(filename_4)

        np.testing.assert_equal(actual_result_1, output_1)
        np.testing.assert_equal(actual_result_2, output_2)
        np.testing.assert_equal(actual_result_3, output_3)
        np.testing.assert_equal(actual_result_4, output_4)




























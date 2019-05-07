from importImage import *
from greenChannel import *
from blueChannelProcessing import *
from sklearn import svm
import logging


def formXYTraining(trainingD, trainingH):

    """ given the total number of training data files,
    generate training matrix for svm

    :param trainingD: the number of diseased training file used
    :param trainingH: the number of healthy training file used
    :type trainingD: String
    :type trainingH: String
    :return: X, Y (np.arrays)
    """

    logging.basicConfig(filename='log.txt', level=logging.DEBUG, format='%(asctime)s %(message)s')

    X = []
    Y = []

    # Read and process diseased data
    for i in range(0, trainingD + 1):
        if len(str(i)) == 1:
            filename = "training/dysplasia0" + str(i) + ".tif"
        else:
            filename = "training/dysplasia" + str(i) + ".tif"
        (red, green, blue) = convertRGB(filename)
        logging.debug(red)
        logging.debug(green)
        logging.debug(blue)
        threshold = otsu(eliminateZero(green))
        logging.info(threshold)
        mode = find_mode(remove_zero(remove_specular(blue)))
        logging.info(mode)
        X.append([threshold, mode])
        Y.append(1)

    # Read and process healthy data
    for j in range(0, trainingH + 1):
        if len(str(j)) == 1:
            filename = "training/healthy0" + str(j) + ".tif"
        else:
            filename = "training/healthy" + str(j) + ".tif"
        (red, green, blue) = convertRGB(filename)
        logging.debug(red)
        logging.debug(green)
        logging.debug(blue)

        threshold = otsu(eliminateZero(green))
        logging.info(threshold)

        mode = find_mode(remove_zero(remove_specular(blue)))
        logging.info(mode)

        X.append([threshold, mode])
        Y.append(0)

    # Data output and storage (caching training params)
    print(X)
    print(Y)
    np.savetxt("training_Xs.txt", X)
    np.savetxt("training_Ys.txt", Y)
    return X, Y


def support_vm(X, Y, params):

    """given training X, Y matrices and and params of the test file,
        provide predicted value

    :param X: training modes and threshold values
    :param Y: training results (0 represents healthy, 1 represents diseased)
    :param params: a matrix of testing values (threshold and mode)
    :type X: numpy array
    :type Y: numpy array
    :returns: clf.predict(params): result of prediction, 1 diseased, 0 healthy
    """


    clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape=None, degree=3, gamma='auto',
                  kernel='rbf', max_iter=-1, probability=False,
                  random_state=None, shrinking=True,
                  tol=0.001, verbose=False)
    clf.fit(X, Y)
    return clf.predict(params)


def diagnosis(filename):
    """ given the testing filename in example folder, diagnose image

    :param filename: name of the test file
    :type filename: String
    :returns: diagnosis(String)
    """

    logging.basicConfig(filename='log.txt', level=logging.DEBUG, format='%(asctime)s %(message)s')

    X = np.loadtxt("training_Xs.txt")
    Y = np.loadtxt("training_Ys.txt")
    (red, green, blue) = convertRGB("example/" + filename)

    logging.debug(red)
    logging.debug(green)
    logging.debug(blue)

    threshold = otsu(eliminateZero(green))
    logging.info(threshold)

    mode = find_mode(remove_zero(remove_specular(blue)))
    logging.info(mode)

    params = []
    params.append([threshold, mode])
    logging.info(params)
    res = support_vm(X, Y, params)

    if res[0] == 1:
        return "diseased"
    else:
        return "healthy"


if __name__ == "__main__":
    # (X, Y) = formXYTraining(5, 5)
    diagnosis = diagnosis("test01.tif")
    print(diagnosis)

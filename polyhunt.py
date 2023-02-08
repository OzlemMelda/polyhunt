"""
author: Ozlem Gunes
"""

# import req libraries
import numpy as np
from sklearn.model_selection import KFold
import argparse


# load and prep data
def loadPrep(trainPath):
    """
    Load data and split it into Train and Test sets.
    :param trainPath: input data path
    :return: Train and Test data and labels.
    """
    # read data
    data = np.loadtxt(trainPath, delimiter=',')
    np.random.shuffle(data)

    # define input variables and output variable
    dataTrain = data.copy()

    # define input variables and output variable
    XTrain = dataTrain[:, 0:1]
    yTrain = dataTrain[:, 1:2]

    return XTrain, yTrain


# build model
def getWeights(X, y, m, gamma):
    """ Basis function: Q(x)=x^m
        Inputs
        X: input variables
        y: output variable
        m:  polynomial order
        Output
        Weight vector"""
    orderList = np.arange(0, m+1, 1)
    designMatrix = np.power(X, orderList)
    nCol = designMatrix.shape[1]
    soln = np.linalg.lstsq(designMatrix.T.dot(designMatrix) + gamma * np.identity(nCol), designMatrix.T.dot(y),
                           rcond=None)
    return soln[0]


# generate predictions
def getPredictions(w, X, m):
    """ Inputs
        X: input variables
        w: calculated weights
        m: polynomial order
        Output
        predictions"""
    orderList = np.arange(0, m+1, 1)
    return np.power(X, orderList).dot(w)


# evaluate model
def evaluate(p, out):
    """
    Evaluates trained model.
    :param p: predictions
    :param out: realized values
    :return: rmse score
    """
    return np.sqrt(np.mean((p-out)**2))


def crossValidation(XTrain, yTrain, i, gamma, numFolds):
    """
    Make cross validation to see if the model is overfitting.
    :param XTrain: train data set
    :param yTrain: train label
    :param i: polynomial order
    :return: rmse error of cross folded data
    """
    kFoldCount = KFold(n_splits=numFolds)
    kfold = kFoldCount.split(XTrain, yTrain)
    scores = []
    for k, (train, test) in enumerate(kfold):
        XTrain_sub = XTrain[train, :]
        yTrain_sub = yTrain[train]
        XTrainTest_sub = XTrain[test, :]
        yTrainTest_sub = yTrain[test]

        w = getWeights(XTrain_sub, yTrain_sub, i, gamma)
        predTest = getPredictions(w, XTrainTest_sub, i)
        score = evaluate(predTest, yTrainTest_sub)
        scores.append(score)
    return scores


def pipeline(XTrain, yTrain, m, gamma, autofit, numFolds):
    """
    Build pipeline to make predictions. Stop when begin overfitting and not enough performance
    :param XTrain: Train data
    :param yTrain: Train label
    :param XTest: Test data
    :param yTest: Test label
    :param m: polynomial order
    :param gamma: regularization parameter
    :param autofit: boolean
    :param numFolds: number of folds
    :return: performance metrics
    """
    # Store metrics in a dict then choose best one
    performanceMetric = {}
    performanceMetric["polynomialOrder"] = []
    performanceMetric["weights"] = []
    performanceMetric["rmse"] = []
    performanceMetric["pred"] = []
    cvMetric = {}
    cvMetric["polynomialOrder"] = []
    cvMetric["cvMean"] = []
    cvMetric["cvStd"] = []
    stopCondition = 0
    if not autofit:
        i = m
    else:
        i = 0
    while i <= m:
        w = getWeights(XTrain, yTrain, i, gamma)
        predTrain = getPredictions(w, XTrain, i)
        rmseTrain = evaluate(predTrain, yTrain)

        performanceMetric["polynomialOrder"].append(i)
        performanceMetric["weights"].append(w)
        performanceMetric["rmse"].append(rmseTrain)
        performanceMetric["pred"].append(predTrain)

        cvScores = crossValidation(XTrain, yTrain, i, gamma, numFolds)
        cvMean = np.mean(cvScores)
        cvStd = np.std(cvScores)
        cvMetric["polynomialOrder"].append(i)
        cvMetric["cvMean"].append(cvMean)
        cvMetric["cvStd"].append(cvStd)

        if len(performanceMetric["polynomialOrder"]) > 1:
            # Calculate change
            meanChange = (cvMetric["cvMean"][i - 1] - cvMetric["cvMean"][i]) / cvMetric["cvMean"][i - 1]
            stdChange = (cvMetric["cvStd"][i - 1] - cvMetric["cvStd"][i]) / cvMetric["cvStd"][i - 1]

            if (meanChange < 0.005) | (stdChange < 0.005):
                stopCondition += 1
        if stopCondition > 5:
            break

        i = i + 1

    return performanceMetric, cvMetric


def getArgument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m',
                        dest='m',
                        type=int)
    parser.add_argument('--gamma',
                        dest='gamma',
                        type=float)
    parser.add_argument('--trainPath',
                        dest='trainPath',
                        type=str)
    parser.add_argument('--modelOutput',
                        dest='modelOutput',
                        type=str)
    parser.add_argument('--autofit',
                        dest='autofit',
                        type=bool)
    parser.add_argument('--info',
                        dest='info',
                        type=bool)
    parser.add_argument('--numFolds',
                        dest='numFolds',
                        type=int)
    return parser.parse_args()


if __name__ == '__main__':

    # init all vars
    maxM = 80
    m = None
    gamma = None
    modelOutput = None
    autofit = None

    # get args from user
    args = getArgument()
    print(args)

    m = args.m
    gamma = args.gamma
    trainPath = args.trainPath
    modelOutput = args.modelOutput
    autofit = args.autofit
    numFolds = args.numFolds

    if (m is None) | (autofit is True):
        m = maxM

    if gamma is None:
        gamma = 0

    if autofit is None:
        autofit = False

    if numFolds is None:
        numFolds = 10

    # get data
    XTrain, yTrain = loadPrep(trainPath)

    # run pipeline
    performanceMetric, cvMetric = pipeline(XTrain, yTrain, m, gamma, autofit, numFolds)

    if autofit is True:
        min_mean = min(performanceMetric["rmse"])
        min_mean_idx = performanceMetric["rmse"].index(min_mean)
        chosen_order = min_mean_idx
        for i in reversed(range(0, min_mean_idx)):
            if performanceMetric["rmse"][i] <= (min_mean*1.05):
                chosen_order = performanceMetric["polynomialOrder"][i]
        chosen_weights = performanceMetric["weights"][chosen_order]
        print("polynomialOrder: " + str(chosen_order))
        print("RMSE: " + str(performanceMetric["rmse"][chosen_order]))
    else:
        chosen_order = performanceMetric["polynomialOrder"][0]
        chosen_weights = performanceMetric["weights"][0]
        print("polynomialOrder: " + str(chosen_order))
        print("RMSE: " + str(performanceMetric["rmse"][0]))

    if modelOutput is not None:
        np.savetxt(modelOutput, chosen_weights, header="m = %d\ngamma = %f" % (chosen_order, gamma))



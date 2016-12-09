import random

import numpy as np

def loadDataSet():
    data_mat = []
    label_mat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        data_mat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        label_mat.append(int(lineArr[2]))
    return data_mat, label_mat


def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelsMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelsMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


def classify_vector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5: return 1.0
    else: return 0.0


def colic_test():
    frTrain = open('data/horseColicTraining.txt')
    frTest = open('data/horseColicTest.txt')
    trainingSet = []; trainingLabels = [];
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        numTestVec += 1.0
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        predicted = int(classify_vector(np.array(lineArr), trainWeights))
        actual = int(currLine[21])
        if predicted != actual:
            errorCount += 1
        errorRate = (float(errorCount) / numTestVec)
        print "the error rate of this test is: %f" % errorRate
        return errorRate


def multiTest():
    numTests = 10; errorSum = 0.0
    for k in range(numTests):
        errorSum += colic_test()
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))


colic_test()
import random
import numpy


def load_data_set(filename):
    data_mat = []; label_mat = []
    fr = open(filename)
    for line in fr.readlines():
        line_arr = line.strip().split('\t')
        data_mat.append([float(line_arr[0]), float(line_arr[1])])
        label_mat.append(float(line_arr[2]))
    return data_mat, label_mat


def selectJrand(i, m):
    j = i
    while(j == i):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = numpy.mat(dataMatIn); labelMat = numpy.mat(classLabels).transpose()
    b = 0
    m, n = numpy.shape(dataMatrix)


data_arr, label_arr = load_data_set('data/svm/testSet.txt')

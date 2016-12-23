from numpy import *


def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)
        dataMat.append(fltLine)
    return dataMat


def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k, 1)
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssignments = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssignments[i, 0] != minIndex:
                clusterChanged = True
            clusterAssignments[i, :] = minIndex, minDist ** 2
        print centroids
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssignments[:,0].A==cent)[0]]
            centroids[cent,:] = mean(ptsInClust, axis = 0)
    return centroids, clusterAssignments


def biKMeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssignments = mat(zeros((m, 2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssignments[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2
    while len(centList) < k:
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrentCluster = dataSet[nonzero(clusterAssignments[:, 0].A == i)[0], :]
            centroidMat, splitClustAss = kMeans(ptsInCurrentCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssignments[nonzero(clusterAssignments[:, 0].A != i)[0], 1])
            print "sseSplit, and notSplit: ", sseSplit, sseNotSplit
            if sseSplit + sseNotSplit < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print 'the bestCentToSplit is: ', bestCentToSplit
        print 'the len of bestClustAss is: ', len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0, :]
        centList.append(bestNewCents[1, :])
        clusterAssignments[nonzero(clusterAssignments[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return mat(centList), clusterAssignments

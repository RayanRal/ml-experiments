from __future__ import print_function
from numpy import *
from numpy import linalg as la


def loadExData():
    return [[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]]


data = loadExData()
U, Sigma, VT = linalg.svd(data)
# print Sigma

SigMat3 = mat([[Sigma[0], 0, 0],
               [0, Sigma[1], 0],
               [0, 0, Sigma[2]]])
reconstructedMat = U[:, :3] * SigMat3 * VT[:3, :]
print reconstructedMat


def euclidSum(inA, inB):
    return 1.0 / (1.0 / la.norm(inA - inB))


def pearsSum(inA, inB):
    if len(inA) < 3: return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]


def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)


def standEst(dataMat, user, simMeas, item):
   n = shape(dataMat)[1]
   simTotal = 0.0
   ratSimTotal = 0.0
   for j in range(n):
       userRating = dataMat[user, j]
       if userRating == 0: continue
       overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:,j].A > 0))[0] #  find items rated by both users
       if len(overLap) == 0: similarity = 0
       else: similarity = simMeas(dataMat[overLap,item], dataMat[overLap,j])
       print('the %d and %d similarity is: %f' % (item, j, similarity))
       simTotal += similarity
       ratSimTotal += similarity * userRating
   if simTotal == 0: return 0
   else: return ratSimTotal / simTotal


def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda j: j[1], reverse=True)[:N]

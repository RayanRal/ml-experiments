import numpy as np
import pandas as pnd
import itertools
from sklearn.svm import SVC

inputData = pnd.read_csv('svmData.csv', index_col=False, header=None)

trainDataTarget = inputData.copy().drop(inputData.columns[1:], axis=1)
trainDataFeatures = inputData.copy().drop(inputData.columns[0], axis=1)

model = SVC(kernel='linear', random_state=241, C=100000)
model.fit(X=trainDataFeatures, y=trainDataTarget)
# testDataPredictions = model.predict(testDataFeatures)

# accuracy = accuracy_score(testDataTarget, testDataPredictions)

print('Support objects: {}').format(model.support_)
    # crossValScore = cross_val_score(estimator=classifier, cv=kFold, scoring='mean_squared_error', X=learnSetNorm, y=dataSet.target)
    # print('for k {} accuracy {}').format(k, crossValScore.max())


# print("classes {}").format(classesNew)
# print("learnSet {}").format(learnSet)
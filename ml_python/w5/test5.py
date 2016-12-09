import numpy as np
import pandas as pnd
import itertools
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

trainData = pnd.read_csv('perceptron-train.csv', index_col=False, header=None)
testData = pnd.read_csv('perceptron-test.csv', index_col=False, header=None)

trainDataTarget = trainData.copy().drop(trainData.columns[1:], axis=1)
trainDataFeatures = trainData.copy().drop(trainData.columns[0], axis=1)

trainDataTargetFlatten = list(itertools.chain(*trainDataTarget.values))

testDataTarget = testData.copy().drop(testData.columns[1:], axis=1)
testDataFeatures = testData.copy().drop(testData.columns[0], axis=1)

trainDataFeaturesScaled = scaler.fit_transform(trainDataFeatures)
testDataFeaturesScaled = scaler.fit_transform(testDataFeatures)

# print('testDataFeatures len: {}').format(testDataFeatures.size)
# print('trainDataTarget: {}').format(trainDataTarget)

model = Perceptron(random_state=241)
model.fit(X=trainDataFeatures, y=trainDataTargetFlatten)
testDataPredictions = model.predict(testDataFeatures)

accuracy = accuracy_score(testDataTarget, testDataPredictions)

print('Accuracy: {}').format(accuracy)
    # crossValScore = cross_val_score(estimator=classifier, cv=kFold, scoring='mean_squared_error', X=learnSetNorm, y=dataSet.target)
    # print('for k {} accuracy {}').format(k, crossValScore.max())


# print("classes {}").format(classesNew)
# print("learnSet {}").format(learnSet)
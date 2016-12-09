import numpy as np
import pandas as pnd
import itertools
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score

inputData = pnd.read_csv('logisticData.csv', index_col=False, header=None)

trainDataTarget = inputData.copy().drop(inputData.columns[1:], axis=1)
trainDataFeatures = inputData.copy().drop(inputData.columns[0], axis=1)

vect = TfidfVectorizer()
trainDataFeaturesTransformed = TfidfVectorizer.fit_transform(vect, raw_documents=trainDataFeatures)
codedWords = vect.get_feature_names()
# print("Feature names: {}").format(vect.get_feature_names())

# grid = {'C': np.power(10.0, np.arange(-5, 6))}
# cv = KFold(trainDataTarget.size, n_folds=5, shuffle=True, random_state=241)
# clf = SVC(kernel='linear', random_state=241)
# gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
# gs.fit(trainDataFeaturesTransformed, trainDataTarget)
# testDataPredictions = model.predict(testDataFeatures)

# accuracy = accuracy_score(testDataTarget, testDataPredictions)

# for a in gs.grid_scores_:
#     print('For a = {} mean validation score: {}, parameters: {}').format(a, a.mean_validation_score, a.parameters)

trainingSvm = SVC(kernel='linear', random_state=241, C=1.0)
trainingSvm.fit(X=trainDataFeaturesTransformed, y=trainDataTarget)
# print("Coefs: {}").format(trainingSvm.coef_)

# maxCoefs = np.argsort(np.absolute(np.asarray(trainingSvm.coef_.todense())).reshape(-1))[-10:]
maxCoefs = pnd.Series(trainingSvm.coef_.toarray().reshape(-1)).abs().nlargest(10).index
print("Max coefs: {}").format(maxCoefs)
for a in maxCoefs:
    print("Word: {}").format(codedWords[a])

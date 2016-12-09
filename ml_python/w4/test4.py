import numpy as np
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale
from sklearn.datasets import load_boston

dataSet = load_boston()

learnSetNorm = scale(dataSet.data)
# print('learnSet size: {}').format(learnSetNorm.size)
# print('target size: {}').format(dataSet.target.size)

kFold = KFold(n=dataSet.target.size, n_folds=5, shuffle=True, random_state=42)

for k in np.linspace(start=1, stop=10, num=200):
    classifier = KNeighborsRegressor(metric='minkowski', p=k, n_neighbors=5, weights='distance')
    crossValScore = cross_val_score(estimator=classifier, cv=kFold, scoring='mean_squared_error', X=learnSetNorm, y=dataSet.target)
    print('for k {} accuracy {}').format(k, crossValScore.max())

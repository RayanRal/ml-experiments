import pandas as pnd
import itertools
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score

input = pnd.read_csv('abalone.csv', index_col=False)
inputTarget = input.copy().drop(input.columns[0:8], axis=1)
inputTargetNew = list(itertools.chain(*inputTarget.values))
inputFeatures = input.copy().drop(input.columns[8], axis=1)

inputFeatures['Sex'] = inputFeatures['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

kFold = KFold(n=inputTarget.size, n_folds=5, shuffle=True, random_state=1)

for k in range(1,51):
    regressor = RandomForestRegressor(random_state=1,n_estimators=k)
    crossValScore = cross_val_score(estimator=regressor, cv=kFold, scoring='r2', X=inputFeatures, y=inputTargetNew)
    print('for k {} accuracy {}').format(k, crossValScore.mean())
import pandas as pnd
import numpy as np
import itertools
import operator
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

data = pnd.read_csv('wine.data.csv', index_col=False, header=None)

classes = data.copy().drop(data.columns[1:], axis=1)
classesNew = list(itertools.chain(*classes.values))
learnSet = data.copy().drop(data.columns[0], axis=1)
learnSetNorm = scale(learnSet)

kFold = KFold(n=classes.size, n_folds=5, shuffle=True, random_state=42)
# print('n: {}').format(classes.size)

for k in range(1,51):
    classifier = KNeighborsClassifier(n_neighbors=k)
    # classifier.fit(learnSet, classesNew)
    crossValScore = cross_val_score(estimator=classifier, cv=kFold, scoring='accuracy', X=learnSetNorm, y=classesNew)
    print('for k {} accuracy {}').format(k, crossValScore.mean())


# print("classes {}").format(classesNew)
# print("learnSet {}").format(learnSet)
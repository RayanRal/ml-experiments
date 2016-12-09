import pandas as pnd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import log_loss
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split

input = pnd.read_csv('gbm-data.csv', index_col=False)
values = input.values
inputTarget = input.copy().drop(input.columns[1:], axis=1)
inputTargetArray = values[:,0]
inputFeaturesArray = values[:,1:]

X_train, X_test, y_train, y_test = train_test_split(inputFeaturesArray, inputTargetArray,test_size=0.8, random_state=241)

n_est = 250
rates = [1, 0.5, 0.3, 0.2, 0.1]
# rates = [0.2]
for k in range(len(rates)):
    print("LEARNING RATE: {}").format(rates[k])
    model = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=rates[k])
    model.fit(X_train, y_train)

    test_score = np.empty(len(model.estimators_))
    train_score = np.empty(len(model.estimators_))

    test_loss = []
    for y_pred in model.staged_decision_function(X_test):
        test_loss.append(log_loss(y_test, 1/(1 + np.exp(-y_pred))))

    for index,element in enumerate(test_loss):
        print ("{0}\t{1}".format(index, element))

model = RandomForestClassifier(random_state=241,n_estimators=36)
model.fit(X_train, y_train)

test_loss = []
y_pred = model.predict_proba(X_test)
test_loss.append(log_loss(y_test, y_pred))

for index,element in enumerate(test_loss):
    print ("{0}\t{1}".format(index, element))

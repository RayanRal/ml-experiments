import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

input_data = pd.read_csv('data/train.csv', header=0, index_col='PassengerId')

test_data = pd.read_csv('data/test.csv', header=0, index_col='PassengerId')

# input_data = input_data.drop('Name', 1).drop('SibSp', 1).drop('Parch', 1).drop('Cabin', 1).drop('Embarked', 1).drop('Ticket', 1)
input_data = input_data.drop('Name', 1).drop('Cabin', 1).drop('Ticket', 1)
input_data['Sex'] = input_data['Sex'].map({'female': 1, 'male': 0})
input_data = pd.get_dummies(input_data, columns=['Embarked'])
input_data.fillna(method='bfill', inplace=True)

test_data = test_data.drop('Name', 1).drop('Cabin', 1).drop('Ticket', 1)
test_data['Sex'] = test_data['Sex'].map({'female': 1, 'male': 0})
test_data = pd.get_dummies(test_data, columns=['Embarked'])
test_data.fillna(method='ffill', inplace=True)

# print input_data
input_classes = input_data.copy().drop('Pclass', 1).drop('Sex', 1).drop('Age', 1).drop('Fare', 1).drop('Parch', 1).drop('Embarked_C', 1).drop('Embarked_Q', 1).drop('Embarked_S', 1).drop('SibSp', 1)
input_data.drop('Survived', 1, inplace=True)
# print test_data
# print input_data
# print input_classes

X_train, X_test, y_train, y_test = train_test_split(input_data, input_classes, test_size=0.2, train_size=0.8, random_state=241)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, criterion='entropy'), algorithm="SAMME", n_estimators=100, random_state=241)
model = RandomForestClassifier(random_state=241, n_estimators=55, criterion='entropy', max_depth=4)
# model = GradientBoostingClassifier(n_estimators=500, verbose=True)
model.fit(X=X_train, y=y_train)
testDataPredictions = model.predict(X_test)

accuracy = accuracy_score(y_test, testDataPredictions)
print 'Accuracy: {}'.format(accuracy)

# print test_data
# test_data['Survived'] = model.predict(test_data)
# test_data = test_data.drop('Pclass', 1).drop('Sex', 1).drop('Age', 1).drop('Fare', 1).drop('Parch', 1).drop('Embarked_C', 1).drop('Embarked_Q', 1).drop('Embarked_S', 1).drop('SibSp', 1)
#
# submission = pd.DataFrame(test_data, columns=['PassengerId', 'Survived'])
# test_data.to_csv('titanic_pred.csv', index=True)

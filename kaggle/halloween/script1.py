import matplotlib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold
from operator import itemgetter
from sklearn.model_selection import GridSearchCV

input_data = pd.read_csv('data/train.csv', header=0, index_col='id')
input_data = input_data.drop('color', 1)  # pd.get_dummies(input_data, columns=['color'])
input_classes = input_data['type']
input_data.drop('type', 1, inplace=True)

# creating additional features
input_data['hair_soul'] = input_data['hair_length'] * input_data['has_soul']
input_data['hair_bone'] = input_data['hair_length'] * input_data['bone_length']
input_data['hair_soul_bone'] = input_data['hair_length'] * input_data['has_soul'] * input_data['bone_length']


# checking feature importance
# X_train, X_test, y_train, y_test = train_test_split(input_data, input_classes, test_size=0.1, train_size=0.9, random_state=241)
# model = RandomForestClassifier(random_state=241, n_estimators=30, criterion='entropy', max_depth=5)
# model.fit(X=X_train, y=y_train)
# indices = np.argsort(model.feature_importances_)[::-1]
# for f in range(X_train.shape[1]):
#     print('%d. feature %d %s (%f)' % (f + 1, indices[f], X_train.columns[indices[f]],
#                                       model.feature_importances_[indices[f]]))


X_train, X_test, y_train, y_test = train_test_split(input_data, input_classes, test_size=0.2, train_size=0.8, random_state=241)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Best parameters: {'multi_class': 'multinomial', 'C': 1000, 'tol': 0.0001, 'solver': 'newton-cg'}
# model = LogisticRegression()
# parameter_grid = {
#     'solver': ['newton-cg', 'lbfgs', 'sag'],
#     'multi_class': ['ovr', 'multinomial'],
#     'C': [0.005, 0.01, 1, 10, 100, 1000],
#     'tol': [0.0001, 0.001, 0.005]
# }
#
# grid_search = GridSearchCV(model, param_grid=parameter_grid, scoring='accuracy', cv=StratifiedKFold(5))
# grid_search.fit(X_train, y_train)
# print('Best score: {}'.format(grid_search.best_score_))
# print('Best parameters: {}'.format(grid_search.best_params_))

model = LogisticRegression(C=1000, tol=0.0001, solver='newton-cg', multi_class='multinomial')
model.fit(X_train, y_train)
testDataPredictions = model.predict(X_test)
accuracy = accuracy_score(y_test, testDataPredictions)
print 'Accuracy: {}'.format(accuracy)


# for depth in range(1, max_depth):
#     for est in range(20, max_estimators, 5):
#         # Best accuracy is 0.7368, with depth 4 and estimators 100, train 0.9, rs = 241, entropy, SAMME
#         # model = AdaBoostClassifier(
#         #     DecisionTreeClassifier(max_depth=depth, criterion='entropy'),
#         #     algorithm="SAMME", n_estimators=est, random_state=241)
#         # Best accuracy is 0.684210526316, with depth 5 and estimators 30
#         model = RandomForestClassifier(random_state=241, n_estimators=est, criterion='entropy', max_depth=depth)
#         model.fit(X=X_train, y=y_train)
#
#         testDataPredictions = model.predict(X_test)
#         accuracy = accuracy_score(y_test, testDataPredictions)
#         options.append((accuracy, depth, est))
#         print 'Accuracy: {}, depth {}, estimators {}'.format(accuracy, depth, est)
#
# bestOptions = max(options, key=itemgetter(0))
# print 'Best accuracy is %(acc)s, with depth %(depth)s and estimators %(est)s' % {'acc': bestOptions[0], 'depth': bestOptions[1], 'est': bestOptions[2]}

# model = GradientBoostingClassifier(n_estimators=1000, verbose=True)

# Best accuracy is 0.7368, with depth 4 and estimators 100, train 0.9, rs = 241, entropy, SAMME
# model = AdaBoostClassifier(
#     DecisionTreeClassifier(max_depth=4, criterion='entropy'),
#     algorithm="SAMME", n_estimators=100, random_state=241)
# model.fit(X=input_data, y=input_classes)
#
test_data = pd.read_csv('data/test.csv', index_col='id')
test_data = test_data.drop('color', 1)
# creating additional features
test_data['hair_soul'] = test_data['hair_length'] * test_data['has_soul']
test_data['hair_bone'] = test_data['hair_length'] * test_data['bone_length']
test_data['hair_soul_bone'] = test_data['hair_length'] * test_data['has_soul'] * test_data['bone_length']
test_data['type'] = model.predict(test_data)
# print test_data
test_data = test_data.drop('bone_length', 1).drop('rotting_flesh', 1).drop('hair_length', 1).drop('has_soul', 1).drop('hair_soul', 1).drop('hair_bone', 1).drop('hair_soul_bone', 1)
test_data.to_csv('monster_pred.csv', index=True)

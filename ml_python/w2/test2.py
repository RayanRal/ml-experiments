import pandas as pnd
import numpy as np
import itertools
import operator
from sklearn.tree import DecisionTreeClassifier

data = pnd.read_csv('titanic.csv', index_col='PassengerId')

# allAmount = data.iloc[:,0].size
# print "All passengers: {}".format(data.iloc[:,0].size)

# print "Men: {}".format(data[data['Sex']=='male'].count())
# print "Women: {}".format(data[data['Sex']=='female'].count())

# survivedAmount = data[data['Survived']==1].iloc[:,0].size
# survivedPercent = (survivedAmount/float(allAmount))*100
# print "Survived percent: {}".format(survivedPercent)

# firstClassAmount = data[data['Pclass']==1].iloc[:,0].size
# fcPercent = (firstClassAmount/float(allAmount))*100
# print "First class percent: {}".format(fcPercent)

# ageMean = data['Age'].mean(axis=0)
# ageMedian = data['Age'].median(axis=0)
# print "Age mean percent: {}".format(ageMean)
# print "Age median: {}".format(ageMedian)

preDf = data.drop('Name', 1).drop('SibSp', 1).drop('Parch', 1).drop('Cabin', 1).drop('Embarked', 1).drop('Ticket', 1)
preDf = preDf.dropna(axis=0)
classifiedDf = preDf.copy().drop('Pclass', 1).drop('Sex', 1).drop('Age', 1).drop('Fare', 1)

learningDf = preDf.copy().drop('Survived', 1)
learningDf['Sex'] = learningDf['Sex'].map({'female': 1, 'male': 0})
# learningDf = learningDf.dropna(axis=0)

# print "id-survived {}".format(classifiedDf)
# print "id-Pclass-Sex-Age-Fare {}".format(learningDf)

clf = DecisionTreeClassifier(random_state=241)
clf.fit(learningDf, classifiedDf)
print "Columns {}".format(learningDf.columns.values)
print "Trained {}".format(clf.feature_importances_)
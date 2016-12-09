import pandas as pnd
import itertools
import operator

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

women = data[data['Sex']=='female']['Name'].values.tolist()

def div(s):
    if('Mrs' not in s):
        return s.split('. ')[1]
    else:
        print s
        return s.split('(')[1]

names = map(div, women)

def div2(s):
    return s.split(' ')[0]

firstNames = map(div2, names)

def most_common(L):
    groups = itertools.groupby(sorted(L))
    def _auxfun((item, iterable)):
        return len(list(iterable)), -L.index(item)
    return max(groups, key=_auxfun)[0]

name = most_common(firstNames)

print "Women: {}".format(name)
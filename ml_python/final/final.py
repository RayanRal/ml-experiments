import numpy as np
import itertools
from sklearn.ensemble import GradientBoostingClassifier
import csv
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas

features = pandas.read_csv('./features_dota.csv', index_col='match_id')

inputFeaturesTest = pandas.read_csv('./features_test.csv', index_col='match_id')

inputTarget = features.copy()[['radiant_win']]
inputTargetNew = list(itertools.chain(*inputTarget.values))
inputFeatures = features.copy().drop(features.columns[-6:], axis=1)

# Total rows in features and target: matchIds: 114406, rows 97230
# Empty rows: first_blood_time, first_blood_team, first_blood_player1, first_blood_player2
# radiant_bottle_time, radiant_courier_time, radiant_flying_courier_time
# dire_bottle_time, dire_courier_time, dire_flying_courier_time
# radiant_first_ward_time, dire_first_ward_time

# Filling na/s
inputFeatures[['first_blood_time', 'first_blood_team', 'first_blood_player1', 'first_blood_player2', 'radiant_bottle_time', 'radiant_courier_time', 'radiant_flying_courier_time']] = inputFeatures[['first_blood_time', 'first_blood_team', 'first_blood_player1', 'first_blood_player2', 'radiant_bottle_time', 'radiant_courier_time', 'radiant_flying_courier_time']].fillna(0)
inputFeatures[['dire_bottle_time', 'dire_courier_time', 'dire_flying_courier_time', 'radiant_first_ward_time', 'dire_first_ward_time']] = inputFeatures[['dire_bottle_time', 'dire_courier_time', 'dire_flying_courier_time', 'radiant_first_ward_time', 'dire_first_ward_time']].fillna(0)

inputFeaturesTest[['first_blood_time', 'first_blood_team', 'first_blood_player1', 'first_blood_player2', 'radiant_bottle_time', 'radiant_courier_time', 'radiant_flying_courier_time']] = inputFeaturesTest[['first_blood_time', 'first_blood_team', 'first_blood_player1', 'first_blood_player2', 'radiant_bottle_time', 'radiant_courier_time', 'radiant_flying_courier_time']].fillna(0)
inputFeaturesTest[['dire_bottle_time', 'dire_courier_time', 'dire_flying_courier_time', 'radiant_first_ward_time', 'dire_first_ward_time']] = inputFeaturesTest[['dire_bottle_time', 'dire_courier_time', 'dire_flying_courier_time', 'radiant_first_ward_time', 'dire_first_ward_time']].fillna(0)
# print("features {}").format(inputFeatures.head)

# kFold = KFold(n=inputTarget.size, n_folds=5, shuffle=True)
# Gradient boosting and cross-validation
# ks = [50]
# for k in range(len(ks)):
#     print('Training {} trees').format(ks[k])
#     model = GradientBoostingClassifier(n_estimators=ks[k], verbose=True, learning_rate=0.5)
#     crossValScore = cross_val_score(estimator=model, cv=kFold, scoring='roc_auc', X=inputFeatures, y=inputTargetNew)
#     print('for k {} trees accuracy {}').format(ks[k], crossValScore.mean())



#  **Logistic regression**

# Scale features:
scaler = StandardScaler()
categorials = ["lobby_type", "r1_hero", "r2_hero", "r3_hero", "r4_hero", "r5_hero", "d1_hero", "d2_hero", "d3_hero", "d4_hero", "d5_hero"]
inputFeaturesNoCat = inputFeatures.drop(categorials, axis=1)
inputFeaturesScaled = scaler.fit_transform(inputFeaturesNoCat)
# Test features
inputFeaturesTestNoCat = inputFeaturesTest.drop(categorials, axis=1)
inputFeaturesTestScaled = scaler.fit_transform(inputFeaturesTestNoCat)

# bag of words for categorials
X_pick = np.zeros((inputFeatures.shape[0], 112))
for i, match_id in enumerate(inputFeatures.index):
    for p in xrange(5):
        X_pick[i, inputFeatures.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, inputFeatures.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1


X_pick_test = np.zeros((inputFeaturesTest.shape[0], 112))
for i, match_id in enumerate(inputFeaturesTest.index):
    for p in xrange(5):
        X_pick[i, inputFeaturesTest.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, inputFeaturesTest.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

# merge bag of words and scaled features
inputDataFeaturesGlued = np.concatenate([inputFeaturesScaled, X_pick], axis=1)

inputDataFeaturesTestGlued = np.concatenate([inputFeaturesTestScaled, X_pick_test], axis=1)

# Logistic regression and cross-validation
# cs = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
cs = [0.1]
# for c in range(len(cs)):
#     print('Regularization {}').format(cs[c])
#     model = LogisticRegression(C=cs[c], penalty='l2')
#     model.fit(X=inputDataFeaturesGlued, y=inputTargetNew)
#     model.predict(inputDataFeaturesTestGlued)
#     crossValScore = cross_val_score(estimator=model, cv=kFold, scoring='roc_auc', X=inputDataFeaturesGlued, y=inputTargetNew)
#     print('for regularization {} accuracy {}').format(cs[c], crossValScore.mean())

print('Regularization {}').format(0.1)
model = LogisticRegression(C=0.1, penalty='l2')
model.fit(X=inputDataFeaturesGlued, y=inputTargetNew)
predictions = model.predict_proba(inputDataFeaturesTestGlued)[:, 1]



results = zip(inputFeaturesTest.index.values, predictions)

print "Predictions {}".format(results)

f = open('dota-out.csv', 'wt')
try:
    writer = csv.writer(f)
    writer.writerow( ('match_id', 'radiant_win') )
    for a in results:
        writer.writerow( (a[0], a[1]) )
finally:
    f.close()
import numpy as np
import pandas as pnd
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

classificationsInput = pnd.read_csv('classifications.csv', index_col=False)

TP = 0
FP = 0
FN = 0
TN = 0

for r in zip(classificationsInput['true'], classificationsInput['pred']):
    # print('input: {}, {}').format(r[0], r[1])
    if r[0] == 1 and r[1] == 1:
        # print 'predicted TP'
        TP += 1
    if r[0] == 0 and r[1] == 0:
        # print 'predicted TN'
        TN += 1
    if r[0] == 0 and r[1] == 1:
        # print 'predicted FP'
        FP += 1
    if r[0] == 1 and r[1] == 0:
        # print 'predicted FN'
        FN += 1

# print('TP {}, FP {}, FN {}, TN {}').format(TP, FP, FN, TN)

# accuracy = metrics.accuracy_score(classificationsInput['true'], classificationsInput['pred'])
# precision = metrics.precision_score(classificationsInput['true'], classificationsInput['pred'])
# recall = metrics.recall_score(classificationsInput['true'], classificationsInput['pred'])
# f1score = metrics.f1_score(classificationsInput['true'], classificationsInput['pred'])

# print('accuracy {}, precision {}, recall {}, f1-score {}').format(accuracy, precision, recall, f1score)
# print('{} {} {} {}').format(accuracy, precision, recall, f1score)

scoresInput = pnd.read_csv('scores.csv', index_col=False)

# logreg = roc_auc_score(scoresInput['true'], scoresInput['score_logreg'])
# svm = roc_auc_score(scoresInput['true'], scoresInput['score_svm'])
# knn = roc_auc_score(scoresInput['true'], scoresInput['score_knn'])
# tree = roc_auc_score(scoresInput['true'], scoresInput['score_tree'])
#
# print('logreg {} svm {} knn {} tree {}').format(logreg, svm, knn, tree)


logreg_precision, logreg_recall, logreg_thresholds = precision_recall_curve(scoresInput['true'], scoresInput['score_logreg'])
svm_precision, svm_recall, svm_thresholds = precision_recall_curve(scoresInput['true'], scoresInput['score_svm'])
knn_precision, knn_recall, knn_thresholds = precision_recall_curve(scoresInput['true'], scoresInput['score_knn'])
tree_precision, tree_recall, tree_thresholds  = precision_recall_curve(scoresInput['true'], scoresInput['score_tree'])

logreg_precisions_filtered = []
for r in zip(logreg_precision, logreg_recall):
    if r[1] >= 0.7:
        logreg_precisions_filtered.append(r[0])

svm_precisions_filtered = []
for r in zip(svm_precision, svm_recall):
    if r[1] >= 0.7:
        svm_precisions_filtered.append(r[0])

knn_precisions_filtered = []
for r in zip(knn_precision, knn_recall):
    if r[1] >= 0.7:
        knn_precisions_filtered.append(r[0])

tree_precisions_filtered = []
for r in zip(tree_precision, tree_recall):
    if r[1] >= 0.7:
        tree_precisions_filtered.append(r[0])


print('logreg_max_precision {}').format(max(logreg_precisions_filtered))
print('svm_max_precision {}').format(max(svm_precisions_filtered))
print('knn_max_precision {}').format(max(knn_precisions_filtered))
print('tree_max_precision {}').format(max(tree_precisions_filtered))
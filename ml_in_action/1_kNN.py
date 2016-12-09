from numpy import *
import operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt


def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


group, labels = create_data_set()
print group


def classify0(inX, dataSet, labels, k):
    datasetSize = dataSet.shape[0]
    diffMat = tile(inX, (datasetSize, 1)) - dataSet
    # distance calculation, euclidian
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()  # distances sorting
    class_count = {}
    for i in range(k):  # we take `k` closest labels
        voteILabel = labels[sortedDistIndicies[i]]  # taking label of i element
        class_count[voteILabel] = class_count.get(voteILabel, 0) + 1  # counting classes
    # take the classCount dictionary and decompose it into a list of tuples
    # and then sort the tuples by the second item in the tuple
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines, 3))
    class_label_vector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        list_from_line = line.split('\t')
        returnMat[index,:] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return returnMat, class_label_vector


def auto_norm(data_set):
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    norm_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - tile(min_vals, (m, 1))  # tile - create matrix same size as arg[2], filled with arg[1]
    norm_data_set = norm_data_set / tile(ranges, (m, 1))
    return norm_data_set, ranges, min_vals


def dating_class_test():
    ho_ratio = 0.10
    dating_data_mat, dating_labels = file2matrix('datingTestSet.txt')
    norm_mat, ranges, min_vals = auto_norm(datingDataMat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * ho_ratio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :], dating_labels[num_test_vecs:m], 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifier_result, dating_labels[i])
        if classifier_result != dating_labels[i]: error_count += 1.0
    print "the total error rate is: %f" % (error_count / float(num_test_vecs))


def img2vector(filename):
    return_vect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vect[0, 32 * i + j] = int(line_str[j])
    return return_vect


def classify_person():
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(raw_input("percentage of time spent playing video games?"))
    ff_miles = float(raw_input("frequent flier miles earned per year?"))
    ice_cream = float(raw_input("liters of ice cream consumed per year?"))
    dating_data_mat, dating_labels = file2matrix('datingTestSet.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    in_arr = array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify0((in_arr - min_vals) / ranges, norm_mat, dating_labels, 3)
    print "You will probably like this person: ", result_list[classifier_result - 1]


def handwriting_class_test():
    hw_labels = []
    training_file_list = listdir('trainingDigits')
    m = len(training_file_list)
    training_mat = zeros((m, 1024))
    for i in range(m):
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])  # pattern class_imgnumber.txt
        hw_labels.append(class_num_str)
        training_mat[i, :] = img2vector('trainingDigits/%s' % file_name_str)
    test_file_list = listdir('testDigits')
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name_str = test_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        vector_under_test = img2vector('testDigits/%s' % file_name_str)
        classifier_result = classify0(vector_under_test, training_mat, hw_labels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifier_result, class_num_str)
        if classifier_result != class_num_str:
            error_count += 1.0
    print "\nthe total number of errors is: %d" % error_count
    print "\nthe total error rate is: %f" % (error_count / float(m_test))


datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
plt.show()
normMat, ranges, minVals = auto_norm(datingDataMat)

import numpy as np


def create_data_set():
    datMat = np.matrix([[1., 2.1],
                        [2., 1.1],
                        [1.3, 1.],
                        [1., 1.],
                        [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
            lineArr =[]
            curLine = line.strip().split('\t')
            for i in range(numFeat-1):
                lineArr.append(float(curLine[i]))
            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def stump_classify(data_matrix, dimen, thresh_val, thresh_ineq):
    return_array = np.ones((np.shape(data_matrix)[0], 1))
    if thresh_ineq == 'lt':
        return_array[data_matrix[:, dimen] <= thresh_val] = -1.0
    else:
        return_array[data_matrix[:, dimen] > thresh_val] = -1.0
    return return_array


def build_stump(data_array, class_labels, D):
    data_matrix = np.mat(data_array)
    label_matrix = np.mat(class_labels).T
    m, n = np.shape(data_matrix)
    num_steps = 10.0
    best_stump = {}
    best_class_est = np.mat(np.zeros((m, 1)))
    min_error = np.inf
    for i in range(n):
        range_min = data_matrix[:, i].min()
        range_max = data_matrix[:, i].max()
        step_size = (range_max - range_min) / num_steps
        for j in range(-1, int(num_steps) + 1):
            for inequal in ['lt', 'gt']:
                thresh_val = (range_min + float(j) * step_size)
                predicted_vals = stump_classify(data_matrix, i, thresh_val, inequal)
                errors_array = np.mat(np.ones((m, 1)))
                errors_array[predicted_vals == label_matrix] = 0
                weighted_error = D.T * errors_array
                print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (
                i, thresh_val, inequal, weighted_error)
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_class_est = predicted_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
        return best_stump, min_error, best_class_est


def adaBoost_train_ds(data_arr, class_labels, num_it=40):
    weak_class_arr = []
    m = np.shape(data_arr)[0]
    D = np.mat(np.ones((m, 1)) / m)
    agg_class_est = np.mat(np.zeros(m, 1))  # class estimate for every data point by every decision stump
    for i in range(num_it):
        best_stump, error, class_est = build_stump(data_arr, class_labels, D)
        print "D: ", D.T
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        best_stump['alpha'] = alpha
        weak_class_arr.append(best_stump)
        print "class_est: ", class_est.T
        expon = np.multiply(-1 * alpha * np.mat(class_labels).T, class_est)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        agg_class_est += alpha * agg_class_est
        print "agg_class_est: ", agg_class_est.T
        agg_errors = np.multiply(np.sign(agg_class_est) != np.mat(class_labels).T, np.ones((m, 1)))
        error_rate = agg_errors.sum() / m
        print "total error: ", error_rate, "\n"
        if error_rate == 0.0:
            break
    return weak_class_arr


def ada_classify(data, classifiers):
    data_matrix = np.mat(data)
    m = np.shape(data_matrix)[0]
    agg_class_est = np.mat(np.zeros((m, 1)))
    for i in range(len(classifiers)):
        class_est = stump_classify(data_matrix, classifiers[i]['dim'], classifiers[i]['thresh'], classifiers[i]['ineq'])
        agg_class_est += classifiers[i]['alpha'] * class_est
        print agg_class_est
    return np.sign(agg_class_est)


def plot_roc(pred_strengths, class_labels):
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)
    ySum = 0.0
    numPosClas = sum(np.array(classLabels) == 1.0)
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = pred_strengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print "the Area Under the Curve is: ", ySum * xStep


datMat, classLabels = create_data_set()

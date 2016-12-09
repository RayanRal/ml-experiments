from math import log

import operator


def calc_shannon_ent(data_set):
    num_entries = len(data_set)
    # creating dict of all possible classes
    label_counts = {}
    for feature_vector in data_set:
        current_label = feature_vector[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1

    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries  # probability of certain label
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def split_data_set(data_set, axis, value):
    return_data_set = []
    for feature_vector in data_set:
        if feature_vector[axis] == value:
            reduced_feature_vector = feature_vector[:axis]
            reduced_feature_vector.extend(feature_vector[axis+1:])
            return_data_set.append(reduced_feature_vector)
    return return_data_set


def choose_best_feature_for_split(data_set):
    num_features = len(data_set[0]) - 1
    base_entropy = calc_shannon_ent(data_set)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):
        # unique list of class labels
        feature_list = [example[i] for example in data_set]  # all possible values for that feature
        unique_vals = set(feature_list)
        new_entropy = 0.0
        # calculate entropy for each split (each possible value of each feature)
        for value in unique_vals:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_count(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(dataset, labels):
    class_list = [example[-1] for example in dataset]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(dataset[0]) == 1:
        return majority_count(class_list)
    best_feature = choose_best_feature_for_split(dataset)
    best_feature_label = labels[best_feature]
    my_tree = {best_feature_label: {}}
    del(labels[best_feature])
    feature_values = [example[best_feature] for example in dataset]
    unique_vals = set(feature_values)
    for value in unique_vals:
        sub_labels = labels[:]
        my_tree[best_feature_label][value] = create_tree(split_data_set(dataset, best_feature, value), sub_labels)
    return my_tree


def classify(input_tree, feature_labels, test_vec):
    first_str = input_tree.keys()[0]
    second_dict = input_tree[first_str]
    feature_index = feature_labels.index(first_str)
    for key in second_dict.keys():
        if test_vec[feature_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feature_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label


def store_tree(input_tree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(input_tree, fw)
    fw.close()


def restore_tree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


def create_data_set():
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']
                ]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


data, labels = create_data_set()
shannon_entropy = calc_shannon_ent(data)
print "shannon entropy of generated dataset is %i" % shannon_entropy

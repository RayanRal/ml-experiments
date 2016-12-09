from numpy import *

def loadDataSet():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return posting_list, class_vec


# adding all words to vocabulary
def create_vocab_list(data_set):
    vocab_set = set([])
    for document in data_set:
        vocab_set = vocab_set | set(document)  # create union of two sets
    return list(vocab_set)


# vector, 0 in all vocabulary words, 1 in places that occur in input phrase
def set_of_words_2_vec(vocab_list, input_set):
    return_vec = [0]*len(vocab_list)  # create a vector of 0
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return return_vec


# vector, 0 in all vocabulary words, numbers in places that occur in input phrase one or more
def bag_of_words_2_vec_mn(vocab_list, input_set):
    return_vec = [0]*len(vocab_list)  # create a vector of 0
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return return_vec


# train_matrix - array of set_of_words_2_vec, for each entry in input
# train_category - array of class labels for each entry in input
def trainNB0(train_matrix, train_category):
    num_train_docs = len(train_matrix)  # size of all training docs
    num_words = len(train_matrix[0])  # size of all vocabulary
    p_abusive = sum(train_category) / float(num_train_docs)
    p_0_num = ones(num_words)
    p_1_num = ones(num_words)
    p_0_denom = 2.0
    p_1_denom = 2.0
    for i in range(num_train_docs):
        if train_category[i] == 1:  # document is abusive
            p_1_num += train_matrix[i]  # tm[i] - vocab with doc marks, p_1_num - vector with all words and amounts of their encounters
            p_1_denom += sum(train_matrix[i])  # number of all words in sentences of that category
        else:
            p_0_num += train_matrix[i]
            p_0_denom += sum(train_matrix[i])
    p_1_vect = log(p_1_num / p_1_denom)
    p_0_vect = log(p_0_num / p_0_denom)
    return p_0_vect, p_1_vect, p_abusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    pClass0 = 1.0 - pClass1
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(pClass0)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = create_vocab_list(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(set_of_words_2_vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(set_of_words_2_vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(set_of_words_2_vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)


def tokenize_string(input_string):
    import re
    tokens = re.compile('\\W*').split(input_string)
    return [token.lower() for token in tokens if len(token) > 2]


def calc_most_freq(vocab_list, full_text):
    import operator
    freq_dict = {}
    for token in vocab_list:
        freq_dict[token] = full_text.count(token)
    sorted_freq = sorted(freq_dict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_freq[:30]


def local_words(feed0, feed1):
    import feedparser
    doc_list = []; class_list = []; full_text = []
    min_len = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(min_len):
        word_list = tokenize_string(feed1['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        word_list = tokenize_string(feed0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    vocab_list = create_vocab_list(doc_list)
    top30_words = calc_most_freq(vocab_list,full_text)
    # remove most frequent words as irrelevant
    for pairW in top30_words:
        if pairW[0] in vocab_list:
            vocab_list.remove(pairW[0])
    training_set = range(2 * min_len)
    test_set = []
    for i in range(20):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del (training_set[rand_index])


def get_top_words(ny, sf):
    import operator
    vocab_list, p0v, p1v = localWords(ny, sf)
    topNY = []; topSF = []
    


testingNB()
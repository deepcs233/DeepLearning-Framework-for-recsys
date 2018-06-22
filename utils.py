import math
import pickle
import time

from conf import PROCESSED_TRAIN_DATA_PATH
from conf import PROCESSED_TEST_DATA_PATH
from conf import PROCESSED_TRAIN_DATA_CACHE_PATH
from conf import PROCESSED_TEST_DATA_CACHE_PATH

def get_train_dataset(cache=True):
    if cache:
        print "Load cache of train dataset"
        s = time.time()
        dataset = pickle.load(open(PROCESSED_TRAIN_DATA_CACHE_PATH, 'rb'))
        print "Load Success, Use Time: ", time.time() - s
        return dataset
    with open(PROCESSED_TRAIN_DATA_PATH) as f:
        dataset = [[int(t) for t in x.strip().split('\t')] for x in f.readlines()]
    # pickle.dump(dataset, open(PROCESSED_TRAIN_DATA_CACHE_PATH, 'wb'))
    return dataset

def get_train_dataset_test(cache=True):
    if cache:
        print "Load cache of train dataset"
        s = time.time()
        dataset = pickle.load(open(PROCESSED_TRAIN_DATA_CACHE_PATH, 'rb'))
        print "Load Success, Use Time: ", time.time() - s
        return dataset
    with open(PROCESSED_TRAIN_DATA_PATH) as f:
        dataset = []
        for line in f:
            if len(dataset) > 10003: break
            dataset.append([int(t) for t in line.strip().split('\t')])
    # pickle.dump(dataset, open(PROCESSED_TRAIN_DATA_CACHE_PATH, 'wb'))
    return dataset

def get_test_dataset(cache=True):
    if cache:
        print "Load cache of test dataset"
        s = time.time()
        dataset = pickle.load(open(PROCESSED_TEST_DATA_CACHE_PATH, 'rb'))
        print "Load Success, Use Time: ", time.time() - s
        return dataset
    with open(PROCESSED_TEST_DATA_PATH) as f:
        dataset = [[int(t) for t in x.strip().split('\t')] for x in f.readlines()]
    pickle.dump(dataset, open(PROCESSED_TEST_DATA_CACHE_PATH, 'wb'))
    return dataset

def sigmoid(num):
    return 1.0 / (1 + math.exp(-num))

def get_log_loss(y_true, y_pred):
    n = len(y_pred)
    scores = []
    for i in range(n):
        score = y_true[i] * math.log(y_pred[i]) + (1 - y_true[i]) * math.log(1 - y_pred[i])
        scores.append(-score)
    return sum(scores) / n

def get_mean_loss(y_true, y_pred):
    n = len(y_pred)
    scores = []
    for i in range(n):
        scores.append(abs(y_pred[i] - y_true[i]))
    return sum(scores) / n

def gen_fid(hash_value, slot_id):
    return (abs(hash(hash_value)) >> 12) + (slot_id << 52)

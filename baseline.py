import pickle
import math
import random # normalvariate
import numpy as np
import time

from conf import PROCESSED_TRAIN_DATA_PATH
from conf import PROCESSED_TEST_DATA_PATH
from utils import get_train_dataset
from utils import get_test_dataset
import utils

ITER_NUM = 6
LEARNING_RATE = 0.0006
C = 0#0.000001

dataset = get_train_dataset(False)
INSTANCE_NUM = len(dataset)
train_dataset = dataset[: int(INSTANCE_NUM * 0.999)]
dev_dataset = dataset[int(INSTANCE_NUM * 0.88):]
train_dataset_num = len(train_dataset)
dev_dataset_num = len(dev_dataset)
test_dataset = get_test_dataset(False)
test_dataset_num = len(test_dataset)
slot_num = len(dataset[0]) - 1
slot_array = [random.normalvariate(0, 00.001) for x in range(1024)]

PS = dict()

def get_pred():
    y_pred = []
    y_true = []
    for i in range(dev_dataset_num):
        w = 0
        for fid in dev_dataset[i][:-1]:
            if fid not in PS:
                PS[fid] = random.normalvariate(0, 0.000003) - 4.0 / len(dev_dataset[i][:-1])
            s = fid >> 52
            w += PS[fid]# * slot_array[s]
        #y_pred.append(0.0188)
        y_pred.append(utils.sigmoid(w))
        y_true.append(dev_dataset[i][-1])
    return y_true, y_pred

print "Start Training ..."
for it in range(ITER_NUM):
    random.shuffle(train_dataset)
    true_collector = []
    pred_collector = []
    for i in range(train_dataset_num):
        if i % 100000 == 0 and i != 0:
            log_loss = utils.get_log_loss(true_collector, pred_collector)
            print 'Iter: %s, num: %s, avg loss: %.5s, avg pred label: %.5s' % (it, i, log_loss, sum(pred_collector) / max(1, len(pred_collector)))
            true_collector = []
            pred_collector = []
        fids = train_dataset[i]
        data = fids[:-1]
        label = fids[-1]
        w = 0
        for fid in data:
            if fid not in PS:
                PS[fid] = random.normalvariate(0, 0.000001) - 4.0 / len(data)
            s = fid >> 52
            w += PS[fid]# * slot_array[s]
        delta = label - utils.sigmoid(w) - C * w
        # print delta, label, utils.sigmoid(w), w
        true_collector.append(label)
        pred_collector.append(utils.sigmoid(w))
        for fid in data:
            s = fid >> 52
            #t = delta * PS[fid] * LEARNING_RATE
            PS[fid] += delta * LEARNING_RATE
            #slot_array[s] += t
    LEARNING_RATE = LEARNING_RATE * 0.7

    ''' val '''
    y_true, y_pred = get_pred()
    log_loss = utils.get_log_loss(y_true, y_pred)
    mean_loss = utils.get_mean_loss(y_true, y_pred)
    print "Iter: %s, valid result: , log loss: %.6s, mean loss: %.6s, avg pred label: %.6s" % (it, log_loss, mean_loss, sum(y_pred) / len(y_pred))



pred = []
instance_ids = []
for i in range(test_dataset_num):
    fids = test_dataset[i]
    instance_id = fids[0]
    w = 0
    for fid in fids[1:]:
        if fid not in PS:
            PS[fid] = random.normalvariate(0, 0.00001)
        w += PS[fid]
    pred.append(utils.sigmoid(w))
    instance_ids.append(instance_id)

with open('res/res_318.txt', 'w') as f:
    f.write('instance_id predicted_score\n')
    for i in range(len(pred)):
        f.write(str(instance_ids[i]) + ' ' + str(pred[i]) + '\n')
print(sum(pred) / len(pred))

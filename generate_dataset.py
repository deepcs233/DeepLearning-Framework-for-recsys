# encoding=utf-8
from collections import Counter
import time
import math
from conf import TRAIN_DATA_PATH_w
from conf import TEST_DATA_PATH_2_w
from conf import PROCESSED_TRAIN_DATA_PATH
from conf import PROCESSED_TEST_DATA_PATH
from utils import gen_fid

#FEATURE_NUM = 25
with open(TRAIN_DATA_PATH_w) as f:
    dataset = [line.strip() for line in f.readlines()]
with open(TEST_DATA_PATH_2_w) as f:
    dataset_test = [line.strip() for line in f.readlines()]

numCounter = Counter()

def combineFeature(fids):
    res = []
    return res
    key = [3, 4, 5, 6, 7, 8, 10, 11, 12, 13]
    p = 100
    for k in key:
        for i in key:
            p += 1
            if i == 1:  # data type: a;b;c
                for t in slots[i].split(';'):
                    line_data.append(gen_fid(str(fids[k]) + t + heads[i], p))
            if i == k or i == 14 or i == 2 or (i in key and i > k):
                continue
            if i == 15:
                tup = time.localtime(int(slots[15]))
                hour = tup[3]
                res.append(gen_fid(str(fids[k]) + str(hour), p))
            elif i == 17:# data type: a:a1;b;c:c2
                for t in slots[i].split(';'):
                    if ':' in t:
                        item_category, item_property = t.split(':', 1)
                        line_data.append(gen_fid(str(fids[k]) + item_category + heads[i], p))
                        line_data.append(gen_fid(str(fids[k]) + item_property + heads[i], p))
                    line_data.append(gen_fid(str(fids[k]) + t + heads[i], p))
            elif i == 22 or i == 23 or i == 24 or i == 26:# data type: 0.4325325
                processed_num = int(math.log(1.00001 - float(slots[i])) * 100)
                line_data.append(gen_fid(str(fids[k]) + str(processed_num) + heads[i], p))
            else:
                res.append(gen_fid(str(fids[k]) + str(fids[i]), p))
    return res

''' process data of train '''
released = []
heads = dataset[0]
tp = []
for line in dataset[1:]:
    line_data = []
    slots = [x for x in line.split() if x][1:]# 去除第一个instance_id, 无意义
    ex_fids = combineFeature(slots[:-1])
    FEATURE_NUM =len(slots)
    for i in range(FEATURE_NUM):
        if i == 1 or i == 2:  # data type: a;b;c
            for t in slots[i].split(';'):
                line_data.append(gen_fid(t + heads[i], i))
        elif i == 6:
            sa_le = int(slots[i])
            if sa_le < 10:
                line_data.append(gen_fid('1', 45))
            elif sa_le > 13:
                line_data.append(gen_fid('2', 45))
            else:
                line_data.append(gen_fid('3', 45))
        elif i == 7:
            co_le = int(slots[i])
            if co_le < 12:
                line_data.append(gen_fid('1', 46))
            elif co_le > 13:
                line_data.append(gen_fid('2', 46))
            else:
                line_data.append(gen_fid('3', 46))
        elif i == 13:
            star =  slots[13]
            if star in ('3000', '3001', '3002'):
                line_data.append(gen_fid('low', 42))
            elif star in ('3003', '3004', '3005'):
                line_data.append(gen_fid('med', 42))
            else:
                line_data.append(gen_fid('high', 42))
        if i == 14 or i == len(slots) - 1:
            continue
        elif i == 10:
            age = slots[10]
            if age in ('-1', '1000', '1001', '1002'):
                line_data.append(gen_fid('young', 41))
            elif age in ('1003', '1004'):
                line_data.append(gen_fid('strong', 41))
            else:
                line_data.append(gen_fid('old', 41))
        if i == 15:
            tup = time.localtime(int(slots[15]))
            hour = tup[3]
            minute = tup[4] / 10
            wday = tup[6]
            tp.append(hour)
            line_data.append(gen_fid('hour' + str(hour), 27))
            line_data.append(gen_fid('minute' + str(minute), 28))
            # line_data.append(gen_fid('wday' + str(wday), 29))
            if hour < 7:
                line_data.append(gen_fid('1', 40))
            elif hour <= 12:
                line_data.append(gen_fid('2', 40))
            elif hour <= 18:
                line_data.append(gen_fid('3', 40))
            else:
                line_data.append(gen_fid('4', 40))
        elif i == 17: # data type: a:a1;b;c:c2
            for t in slots[i].split(';'):
                if ':' in t:
                    item_category, item_property = t.split(':', 1)
                    line_data.append(gen_fid(item_category + heads[i], i))
                    line_data.append(gen_fid(item_property + heads[i], i))
                line_data.append(gen_fid(t + heads[i], i))
        elif i == 19:
            com_num = int(slots[19])
            if com_num < 14:
                line_data.append(gen_fid('1', 43))
            elif com_num > 18:
                line_data.append(gen_fid('2', 43))
            else:
                line_data.append(gen_fid('3', 43))
        elif i == 21:
            star_le = int(slots[i])
            if star_le < 5012:
                line_data.append(gen_fid('1', 44))
            elif star_le > 5015:
                line_data.append(gen_fid('2', 44))
            else:
                line_data.append(gen_fid('3', 44))
        elif i == 20 or i == 22 or i == 23 or i == 24:# data type: 0.4325325
            processed_num = int(math.log(1.00001 - float(slots[i])) * 100)
            processed_num_l1 = int(math.log(1.00001 - float(slots[i])) * 10)
            processed_num_l2 = int(float(slots[i]) * 20)
            numCounter[processed_num] += 1
            line_data.append(gen_fid(str(processed_num) + heads[i], i))
            line_data.append(gen_fid(str(processed_num_l1) + heads[i], i + 30))
            line_data.append(gen_fid(str(processed_num_l2) + heads[i], i + 35))
        elif i >= 25 and i <= 32:
            p = int(math.log(int(slots[i]) + 1, 10))
            line_data.append(gen_fid(str(p), i))
        else:
            line_data.append(gen_fid(slots[i] + heads[i], i))
    line_data.extend(ex_fids)
    line_data.append(slots[-1]) # 最后一列为is_trade, 属于label，不需要做离散化处理
    released.append(line_data)

with open(PROCESSED_TRAIN_DATA_PATH, 'w') as f:
    for line in released:
        line = [str(x) for x in line if x]
        f.write('\t'.join(line) + '\n')

''' process data of test '''
released = []
heads = dataset_test[0]

for line in dataset_test[1:]:
    line_data = []
    instance_id = [x for x in line.split() if x][0]
    line_data.append(instance_id)
    slots = [x for x in line.split() if x][1:]# 去除第一个instance_id, 无意义
    ex_fids = combineFeature(slots)
    FEATURE_NUM =len(slots)
    for i in range(FEATURE_NUM):
        if i == 1 or i == 2:  # data type: a;b;c
            for t in slots[i].split(';'):
                line_data.append(gen_fid(t + heads[i], i))
        elif i == 6:
            sa_le = int(slots[i])
            if sa_le < 10:
                line_data.append(gen_fid('1', 45))
            elif sa_le > 13:
                line_data.append(gen_fid('2', 45))
            else:
                line_data.append(gen_fid('3', 45))
        elif i == 7:
            co_le = int(slots[i])
            if co_le < 12:
                line_data.append(gen_fid('1', 46))
            elif co_le > 13:
                line_data.append(gen_fid('2', 46))
            else:
                line_data.append(gen_fid('3', 46))
        elif i == 13:
            star =  slots[13]
            if star in ('3000', '3001', '3002'):
                line_data.append(gen_fid('low', 42))
            elif star in ('3003', '3004', '3005'):
                line_data.append(gen_fid('med', 42))
            else:
                line_data.append(gen_fid('high', 42))
        if i == 14:
            continue
        elif i == 10:
            age = slots[10]
            if age in ('-1', '1000', '1001', '1002'):
                line_data.append(gen_fid('young', 41))
            elif age in ('1003', '1004'):
                line_data.append(gen_fid('strong', 41))
            else:
                line_data.append(gen_fid('old', 41))
        if i == 15:
            tup = time.localtime(int(slots[15]))
            hour = tup[3]
            minute = tup[4] / 10
            wday = tup[6]
            tp.append(hour)
            line_data.append(gen_fid('hour' + str(hour), 27))
            line_data.append(gen_fid('minute' + str(minute), 28))
            # line_data.append(gen_fid('wday' + str(wday), 29))
            if hour < 7:
                line_data.append(gen_fid('1', 40))
            elif hour <= 12:
                line_data.append(gen_fid('2', 40))
            elif hour <= 18:
                line_data.append(gen_fid('3', 40))
            else:
                line_data.append(gen_fid('4', 40))
        elif i == 17: # data type: a:a1;b;c:c2
            for t in slots[i].split(';'):
                if ':' in t:
                    item_category, item_property = t.split(':', 1)
                    line_data.append(gen_fid(item_category + heads[i], i))
                    line_data.append(gen_fid(item_property + heads[i], i))
                line_data.append(gen_fid(t + heads[i], i))
        elif i == 19:
            com_num = int(slots[19])
            if com_num < 14:
                line_data.append(gen_fid('1', 43))
            elif com_num > 18:
                line_data.append(gen_fid('2', 43))
            else:
                line_data.append(gen_fid('3', 43))
        elif i == 21:
            star_le = int(slots[i])
            if star_le < 5012:
                line_data.append(gen_fid('1', 44))
            elif star_le > 5015:
                line_data.append(gen_fid('2', 44))
            else:
                line_data.append(gen_fid('3', 44))
        elif i == 20 or i == 22 or i == 23 or i == 24:# data type: 0.4325325
            processed_num = int(math.log(1.00001 - float(slots[i])) * 100)
            processed_num_l1 = int(math.log(1.00001 - float(slots[i])) * 10)
            processed_num_l2 = int(float(slots[i]) * 20)
            numCounter[processed_num] += 1
            line_data.append(gen_fid(str(processed_num) + heads[i], i))
            line_data.append(gen_fid(str(processed_num_l1) + heads[i], i + 30))
            line_data.append(gen_fid(str(processed_num_l2) + heads[i], i + 35))
        elif i >= 25 and i <= 32:
            p = int(math.log(int(slots[i]) + 1, 10))
            line_data.append(gen_fid(str(p), i))
        else:
            line_data.append(gen_fid(slots[i] + heads[i], i))
    released.append(line_data)

with open(PROCESSED_TEST_DATA_PATH, 'w') as f:
    for line in released:
        line = [str(x) for x in line if x]
        f.write('\t'.join(line) + '\n')


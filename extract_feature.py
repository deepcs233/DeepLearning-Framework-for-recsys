TRAIN_DATA_PATH = 'data/processes_train_time.tsv'
TEST_DATA_PATH_1 = 'data/round1_ijcai_18_test_a_20180301.txt'
TEST_DATA_PATH_2 = 'data/round1_ijcai_18_test_b_20180418.txt'

TRAIN_DATA_PATH_w = 'data/round1_ijcai_18_train_20180301_t.txt'
TEST_DATA_PATH_1_w = 'data/round1_ijcai_18_test_a_20180301_t.txt'
TEST_DATA_PATH_2_w = 'data/round1_ijcai_18_test_b_20180418_t.txt'
from collections import defaultdict

in_user = defaultdict(list)
in_user_item = defaultdict(list)
in_shop_item = defaultdict(list)
in_user_shop = defaultdict(list)

with open(TRAIN_DATA_PATH) as f:
    lines = f.readlines()[1: ]
    for line in lines:
        line = line.strip().split()
        in_id = line[0]
        item_id = line[1]
        user_id = line[10]
        shop_id = line[19]
        timestamp = int(line[16])
        in_user[user_id].append(timestamp)
        in_user_item[user_id + item_id].append(timestamp)
        in_shop_item[shop_id + item_id].append(timestamp)
        in_user_shop[user_id + shop_id].append(timestamp)
print("step 1")

with open(TEST_DATA_PATH_1) as f:
    lines = f.readlines()[1: ]
    for line in lines:
        line = line.strip().split()
        in_id = line[0]
        item_id = line[1]
        user_id = line[10]
        shop_id = line[19]
        timestamp = int(line[16])
        in_user[user_id].append(timestamp)
        in_user_item[user_id + item_id].append(timestamp)
        in_shop_item[shop_id + item_id].append(timestamp)
        in_user_shop[user_id + shop_id].append(timestamp)

print("step 2")
with open(TEST_DATA_PATH_2) as f:
    lines = f.readlines()[1: ]
    for line in lines:
        line = line.strip().split()
        in_id = line[0]
        item_id = line[1]
        user_id = line[10]
        shop_id = line[19]
        timestamp = int(line[16])
        in_user[user_id].append(timestamp)
        in_user_item[user_id + item_id].append(timestamp)
        in_shop_item[shop_id + item_id].append(timestamp)
        in_user_shop[user_id + shop_id].append(timestamp)

import bisect

print("step 3")
with open(TRAIN_DATA_PATH_w, 'w') as f_w:
    with open(TRAIN_DATA_PATH) as f:
        lines = f.readlines()
        f_w.write(' '.join(lines[0].strip().split()[:-1]) + ' ' + ' '.join(['u_b', 'u_a', 'ui_b', 'ui_a', 'si_b', 'si_a', 'us_b', 'us_a']) + ' is_trade' + '\n')
        for line in lines[1: ]:
            ex = []
            line = line.strip().split()
            in_id = line[0]
            item_id = line[1]
            user_id = line[10]
            shop_id = line[19]
            timestamp = int(line[16])
            if len(in_user[user_id]) > 1:
                in_user[user_id].sort()
                index = bisect.bisect_left(in_user[user_id], timestamp)
                if index == 0:
                    ex.append(10000000)
                else:
                    ex.append(timestamp - in_user[user_id][index - 1])
                if index >= len(in_user[user_id]) - 1:
                    ex.append(10000000)
                else:
                    ex.append(in_user[user_id][index + 1] - timestamp)
            else:
                ex.extend([10000000, 10000000])
            if len(in_user_item[user_id + item_id]) > 1:
                in_user_item[user_id + item_id].sort()
                index = bisect.bisect_left(in_user_item[user_id + item_id], timestamp)
                if index == 0:
                    ex.append(10000000)
                else:
                    ex.append(timestamp - in_user_item[user_id + item_id][index - 1])
                if index >= len(in_user_item[user_id]) - 1:
                    ex.append(10000000)
                else:
                    ex.append(in_user_item[user_id + item_id][index + 1] - timestamp)
            else:
                ex.extend([10000000, 10000000])
            if len(in_shop_item[user_id + item_id]) > 1:
                in_shop_item[user_id + item_id].sort()
                index = bisect.bisect_left(in_shop_item[user_id + item_id], timestamp)
                if index == 0:
                    ex.append(10000000)
                else:
                    ex.append(timestamp - in_shop_item[user_id + item_id][index - 1])
                if index >= len(in_shop_item[user_id + item_id]) - 1:
                    ex.append(10000000)
                else:
                    ex.append(in_shop_item[user_id + item_id][index + 1] - timestamp)
            else:
                ex.extend([10000000, 10000000])
            if len(in_user_shop[user_id + shop_id]) > 1:
                print len(in_user_shop[user_id + shop_id])
                in_user_shop[user_id + shop_id].sort()
                index = bisect.bisect_left(in_user_shop[user_id + shop_id], timestamp)
                if index == 0:
                    ex.append(10000000)
                else:
                    ex.append(timestamp - in_user_shop[user_id + shop_id][index - 1])
                if index >= len(in_user_shop[user_id + shop_id]) - 1:
                    ex.append(10000000)
                else:
                    ex.append(in_user_shop[user_id + shop_id][index + 1] - timestamp)
            else:
                ex.extend([10000000, 10000000])
            ex = [str(x) for x in ex]
            line = line[:-1] + ex + [line[-1]]
            f_w.write(' '.join(line) + '\n')

print("step 4")
with open(TEST_DATA_PATH_1_w, 'w') as f_w:
    with open(TEST_DATA_PATH_1) as f:
        lines = f.readlines()
        f_w.write(' '.join(lines[0].strip().split()) + ' ' + ' '.join(['u_b', 'u_a', 'ui_b', 'ui_a', 'si_b', 'si_a', 'us_b', 'us_a']) + '\n')
        for line in lines[1: ]:
            ex = []
            line = line.strip().split()
            in_id = line[0]
            item_id = line[1]
            user_id = line[10]
            shop_id = line[19]
            timestamp = int(line[16])
            if len(in_user[user_id]) > 1:
                in_user[user_id].sort()
                index = bisect.bisect_left(in_user[user_id], timestamp)
                if index == 0:
                    ex.append(10000000)
                else:
                    ex.append(timestamp - in_user[user_id][index - 1])
                if index >= len(in_user[user_id]) - 1:
                    ex.append(10000000)
                else:
                    ex.append(in_user[user_id][index + 1] - timestamp)
            else:
                ex.extend([10000000, 10000000])
            if len(in_user_item[user_id + item_id]) > 1:
                in_user_item[user_id + item_id].sort()
                index = bisect.bisect_left(in_user_item[user_id + item_id], timestamp)
                if index == 0:
                    ex.append(10000000)
                else:
                    ex.append(timestamp - in_user_item[user_id + item_id][index - 1])
                if index >= len(in_user_item[user_id]) - 1:
                    ex.append(10000000)
                else:
                    ex.append(in_user_item[user_id + item_id][index + 1] - timestamp)
            else:
                ex.extend([0, 0])
            if len(in_shop_item[user_id + item_id]) > 1:
                in_shop_item[user_id + item_id].sort()
                index = bisect.bisect_left(in_shop_item[user_id + item_id], timestamp)
                if index == 0:
                    ex.append(10000000)
                else:
                    ex.append(timestamp - in_shop_item[user_id + item_id][index - 1])
                if index >= len(in_shop_item[user_id + item_id]) - 1:
                    ex.append(10000000)
                else:
                    ex.append(in_shop_item[user_id + item_id][index + 1] - timestamp)
            else:
                ex.extend([10000000, 10000000])
            if len(in_user_shop[user_id + shop_id]) > 1:
                in_user_shop[user_id + shop_id].sort()
                index = bisect.bisect_left(in_user_shop[user_id + shop_id], timestamp)
                if index == 0:
                    ex.append(10000000)
                else:
                    ex.append(timestamp - in_user_shop[user_id + shop_id][index - 1])
                if index >= len(in_user_shop[user_id + shop_id]) - 1:
                    ex.append(10000000)
                else:
                    ex.append(in_user_shop[user_id + shop_id][index + 1] - timestamp)
            else:
                ex.extend([10000000, 10000000])
            ex = [str(x) for x in ex]
            line = line + ex
            f_w.write(' '.join(line) + '\n')

print("step 5")
with open(TEST_DATA_PATH_2_w, 'w') as f_w:
    with open(TEST_DATA_PATH_2) as f:
        lines = f.readlines()
        f_w.write(' '.join(lines[0].strip().split()) + ' ' + ' '.join(['u_b', 'u_a', 'ui_b', 'ui_a', 'si_b', 'si_a', 'us_b', 'us_a']) + '\n')
        for line in lines[1: ]:
            ex = []
            line = line.strip().split()
            in_id = line[0]
            item_id = line[1]
            user_id = line[10]
            shop_id = line[19]
            timestamp = int(line[16])
            if len(in_user[user_id]) > 1:
                in_user[user_id].sort()
                index = bisect.bisect_left(in_user[user_id], timestamp)
                if index == 0:
                    ex.append(10000000)
                else:
                    ex.append(timestamp - in_user[user_id][index - 1])
                if index >= len(in_user[user_id]) - 1:
                    ex.append(10000000)
                else:
                    ex.append(in_user[user_id][index + 1] - timestamp)
            else:
                ex.extend([10000000, 10000000])
            if len(in_user_item[user_id + item_id]) > 1:
                in_user_item[user_id + item_id].sort()
                index = bisect.bisect_left(in_user_item[user_id + item_id], timestamp)
                if index == 0:
                    ex.append(10000000)
                else:
                    ex.append(timestamp - in_user_item[user_id + item_id][index - 1])
                if index >= len(in_user_item[user_id]) - 1:
                    ex.append(10000000)
                else:
                    ex.append(in_user_item[user_id + item_id][index + 1] - timestamp)
            else:
                ex.extend([10000000, 10000000])
            if len(in_shop_item[user_id + item_id]) > 1:
                in_shop_item[user_id + item_id].sort()
                index = bisect.bisect_left(in_shop_item[user_id + item_id], timestamp)
                if index == 0:
                    ex.append(10000000)
                else:
                    ex.append(timestamp - in_shop_item[user_id + item_id][index - 1])
                if index >= len(in_shop_item[user_id + item_id]) - 1:
                    ex.append(10000000)
                else:
                    ex.append(in_shop_item[user_id + item_id][index + 1] - timestamp)
            else:
                ex.extend([10000000, 10000000])
            if len(in_user_shop[user_id + shop_id]) > 1:
                in_user_shop[user_id + shop_id].sort()
                index = bisect.bisect_left(in_user_shop[user_id + shop_id], timestamp)
                if index == 0:
                    ex.append(10000000)
                else:
                    ex.append(timestamp - in_user_shop[user_id + shop_id][index - 1])
                if index >= len(in_user_shop[user_id + shop_id]) - 1:
                    ex.append(10000000)
                else:
                    ex.append(in_user_shop[user_id + shop_id][index + 1] - timestamp)
            else:
                ex.extend([10000000, 10000000])
            ex = [str(x) for x in ex]
            line = line + ex
            f_w.write(' '.join(line) + '\n')


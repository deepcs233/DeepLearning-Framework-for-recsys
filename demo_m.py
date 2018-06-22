# encoding=utf-8

import math
import random # normalvariate
import numpy as np
import time

from src.ioManger import ioManger
from src.ioManger import MutiIoManger
from src.ioManger import Vec
from src.layers import DotLayer
from src.layers import SigmoidLayer
from src.layers import FMGroupLayer
from src.layers import Node
from src.layers import RowPoolingLayer
from src.net import Net
from src.model import Model
from src.optimizer import SGD
from src.loss import cross_entropy_error

from conf import PROCESSED_TRAIN_DATA_PATH
from conf import PROCESSED_TEST_DATA_PATH
from utils import get_train_dataset
from utils import get_train_dataset_test
from utils import get_test_dataset
import utils


dataset = get_train_dataset_test(False)
X = [line[:-1] for line in dataset]
y = [line[-1] for line in dataset]
# PS = dict()
iM = MutiIoManger(batch_size=16, read_process_num=1)

net = Net(iM)
# TODO 无头检测

# User - Item, Shop, Contenxt
fm_config = [[['slot_0', 'slot_1', 'slot_2', 'slot_3', 'slot_4', 'slot_5', 'slot_6', 'slot_7', 'slot_8', 'slot_16', 'slot_17', 'slot_18',
               'slot_19', 'slot_20', 'slot_21', 'slot_22', 'slot_23', 'slot_24', 'slot_27', 'slot_28'], ['slot_9', 'slot_10', 'slot_11', 'slot_12', 'slot_13'], 16]]

# Item - Context
fm_config.append([['slot_0', 'slot_1', 'slot_2', 'slot_3', 'slot_4', 'slot_5', 'slot_6', 'slot_7', 'slot_8'], ['slot_27', 'slot_28', 'slot_16'], 8])

# Item - Shop
fm_config.append([['slot_0', 'slot_1', 'slot_2', 'slot_3', 'slot_4', 'slot_5', 'slot_6', 'slot_7', 'slot_8'], ['slot_18', 'slot_19', 'slot_20', 'slot_21', 'slot_22', 'slot_23', 'slot_24'], 8])

# Context - Shop
fm_config.append([['slot_18', 'slot_19', 'slot_20', 'slot_21', 'slot_22', 'slot_23', 'slot_24'], ['slot_27', 'slot_28', 'slot_16'], 8])

# net.add(FMGroupLayer(inputs=['Input'], outputs=['med'], fm_config=fm_config))
# net.add(DotLayer(inputs=['bias_net'], outputs=['aa'], hiddenNum=1))
# net.add(RowPoolingLayer(inputs=['med'], outputs=['aa']))
net.add(RowPoolingLayer(inputs=['bias_net'], outputs=['aa']))
# net.add(DotLayer(inputs=['med'], outputs=['aa'], hiddenNum=1))
net.add(SigmoidLayer(inputs=['aa'], outputs=['Output']))
sgd = SGD(lr=0.00003)#,momentum=0.99,nesterov=True)

model = Model(net=net, optimizer=sgd, lossFunc=cross_entropy_error(), ioManger=iM)
model.build()
model.fit(X=X, y=y, validation_split=0.12, epochs=1)

test_dataset = get_test_dataset(False)
pred = []
val_data = []
instance_ids = []
for i in range(len(test_dataset)):
    fids = test_dataset[i]
    instance_id = fids[0]
    instance_ids.append(instance_id)
    val_data.append(fids[1:])
pred = model.predict(val_data)
print len(pred), len(instance_ids)
with open('res/res_322.txt', 'w') as f:
    f.write('instance_id predicted_score\n')
    for i in range(len(pred)):
        f.write(str(instance_ids[i]) + ' ' + str(pred[i]) + '\n')
print(sum(pred) / len(pred))

# encoding=utf-8

import random
import ConfigParser
import collections
from multiprocessing import Process, Queue
import multiprocessing
import copy
import time

import numpy as np


class ioManger(object):
    def __init__(self, PS=None, batch_size=1, **kwargs):
        self.bolbs = []
        self.config= kwargs
        self.scale = kwargs.get('scale', 0.01)
        self.slot_id_range = kwargs.get('slot_id_range', (0,  1024))
        self.vec_length = {}
        #for slot_id in range(self.slot_id_range[0], self.slot_id_range[1]):
        #    self.vec_length[slot_id] = 0
        self.vec_array = {}
        self.iter_num = -1
        ''' dict[slot] = [fid1, fid2, fid3...] '''
        if PS is None:
            PS = dict()
        self.PS = PS
        self.slot_to_fids = collections.defaultdict(list)
        self.batch_size = batch_size
        self.bias = None


    def apply(self, slot_id, length):
        self.vec_length.setdefault(slot_id, 0)
        offset = self.vec_length[slot_id]
        self.vec_length[slot_id] += length
        blob = Blob(slot_id=slot_id, offset=offset, length=length,
                    batch_size=self.batch_size)
        self.bolbs.append(blob)
        return blob

    def initArray(self, scale):
        '''
        this func should be called after all vec applying space
        '''
        for slot in self.vec_length:
            array = np.random.normal(0, scale,
                                     (self.batch_size, self.vec_length[slot]))
            self.vec_array[slot] = array
        for blob in self.bolbs:
            blob.setArray(self.vec_array[blob.slot_id])

    def initBias(self):
        if self.bias is None:
            self.bias = BiasNet(batch_size=self.batch_size, config=self.config)
        return self.bias

    def clear(self):
        for slot in self.vec_array:
            self.vec_array[slot].fill(0.0)
        for blob in self.bolbs:
            blob.clear()
        if self.bias:
            self.bias.clear()
        self.slot_to_fids = {i: collections.defaultdict(list) for i in range(self.batch_size)}

    def update(self):
        for i in range(self.batch_size):
            if self.bias:
                for slot_id in self.slot_to_fids[i]:
                    for fid in self.slot_to_fids[i][slot_id]:
                        self.PS[fid] += self.bias.getValue(slot_id, i) * 100
            for slot_id in self.vec_array:
                for fid in self.slot_to_fids[i][slot_id]:
                    v_fid = fid + (1 << 62)
                    self.PS[v_fid] += self.vec_array[slot_id][i, :] * 100

    def sentData(self, train_data, train_labels=None):
        '''
        如果train_labels=None, 则为predict模式
        '''
        self.train_data = train_data
        self.train_labels = train_labels
        self.train_data_offset = 0
        self.iter_num = 0
        self.train_data_num = len(self.train_data)
        self.clear()

    def readInstance(self):
        '''
        self.scale 需要根据config 调整
        如果剩下的小于batch_size，应该min()
        v_fid: vector fid
        '''
        self.clear()
        step_offset = min(self.train_data_num - self.train_data_offset,
                          self.batch_size)
        if self.train_labels:
            labels = self.train_labels[self.train_data_offset: self.train_data_offset + step_offset]
            labels.extend([0] * (self.batch_size - step_offset))
            labels = np.array(labels, ndmin=2).T
        if step_offset == 0:
            step_offset = self.batch_size
        if self.train_data_offset > self.train_data_num - self.batch_size:
            self.train_data_offset = 0
            self.iter_num += 1
        for i in range(step_offset):
            instance = self.train_data[self.train_data_offset]
            for fid in instance:
                slot_id = (fid >> 52) & 1023
                v_fid = fid + (1 << 62)
                if v_fid not in self.PS and slot_id in self.vec_length:
                    self.PS[v_fid] = np.random.normal(0, self.scale, (self.vec_length[slot_id]))
                if fid not in self.PS:
                    self.PS[fid] = random.normalvariate(0, self.scale)
                if slot_id in self.vec_length:
                    self.vec_array[slot_id][i, :] += self.PS[v_fid]
                if self.bias:
                    self.bias.addValue(slot_id, self.PS[fid], i)
                self.slot_to_fids[i][slot_id].append(fid)
            self.train_data_offset += 1

        if self.train_labels:
            return labels



class BiasNet():
    def __init__(self, batch_size=1, config={}):
        self.config = config
        self.bias_slots = config.get('bias_slots', [i for i in range(0, 1024)])# need do something
        self.value = np.zeros((batch_size, len(self.bias_slots)))
        self.output_size = self.value.shape
        self.type = 'input'
        self.gradient = None
        self.map_of_slot = {}
        self.inbound_layers = []
        self.outbound_layers = []
        self.name = 'bias_net'
        self.batch_size = batch_size
        for i in range(len(self.bias_slots)):
            self.map_of_slot[self.bias_slots[i]] = i

    def addValue(self, slot_id, add_value, batch_index=None):
        if batch_index is None:
            self.value[:, self.map_of_slot[slot_id]] += add_value
        else:
            self.value[batch_index, self.map_of_slot[slot_id]] += add_value

    def getValue(self, slot_id, batch_index=None):
        if batch_index is None:
            return self.value[:, self.map_of_slot[slot_id]]
        else:
            return self.value[batch_index, self.map_of_slot[slot_id]]

    def receiveBackwardValue(self, gradient):
        if gradient.shape != self.output_size:
            raise Exception('gradient shape not equal value shape!\n'
                            'gradient\'s shape is %s, self.output_size\''
                            'shape is %s'
                             % (str(gradient.shape), str(self.output_size)))
        if self.gradient is None:
            self.gradient = gradient
        else:
            self.gradient += gradient

    def clear(self):
        self.value.fill(0.0)
        self.gradient = None

    def update(self, updateDelta):
        self.value = updateDelta

    @property
    def value(self):
        self.gradient = None
        return self.value


class Blob():
    def __init__(self, slot_id, offset, length, batch_size, config=None, name=None):
        self.slot_id = slot_id
        self.offset = offset
        self.length = length
        self.value = None
        self.type = 'input'
        self.gradient = None
        self.inbound_layers = []
        self.outbound_layers = []
        self.batch_size = batch_size
        self.output_size = (batch_size, length)
        if name is None:
            self.name = str(slot_id) + ':' + str(offset)

    def setArray(self, array):
        self.value = array[:, self.offset: self.offset + self.length]

    def receiveBackwardValue(self, gradient):
        if gradient.shape != self.output_size:
            raise Exception('gradient shape not equal value shape!\n'
                            'gradient\'s shape is %s, self.output_size\''
                            'shape is %s'
                             % (str(gradient.shape), str(self.output_size)))
        if self.gradient is None:
            self.gradient = gradient
        else:
            self.gradient += gradient

    def update(self, updateDelta):
        self.value[:] = updateDelta

    @property
    def value(self):
        self.gradient = None
        return self.value

    def clear(self):
        self.value.fill(0.0)
        self.gradient = None

    def __repr__(self):
        return '[Blob]: slot_id:offset ' + self.name


class Vec():
    def __init__(self, slot_id, ioManger):
        self.slot_id = slot_id
        self.iM = ioManger

    def cross(self, vec2, length):
        blob_1 = self.iM.apply(self.slot_id, length)
        blob_2 = self.iM.apply(vec2.slot_id, length)
        return blob_1, blob_2

    def pureApply(self, length):
        blob = ioManger.apply(self.slot_id, length)
        return blob

# ----------- Muti Process Part ------------

class MutiIoManger(ioManger):
    def __init__(self, PS=None, batch_size=1, read_process_num=1,
                 update_process_num=1, **kwargs):
        super(MutiIoManger, self).__init__(PS=None, batch_size=batch_size, **kwargs)
        self.read_process_num = read_process_num
        self.update_process_num = update_process_num
        self.isMutiProcessInit = False
        self.buffer_maxsize = kwargs.get('buffer_maxsize', 10)
        self.mgr = multiprocessing.Manager()
        self.buffer_queue = self.mgr.Queue(self.buffer_maxsize)
        self.PS = self.mgr.dict()
        self.ReaderInstances = []

    def sentData(self, train_data, train_labels=None):
        if len(self.ReaderInstances) > 0:
            for Reader in self.ReaderInstances:
                if Reader.is_alive():
                    Reader.terminate()
        self.train_data = train_data
        self.train_data_offset = 0
        self.iter_num = 0
        self.buffer_size = 0
        self.train_data_num = len(self.train_data)
        self.ReaderInstances = []
        if not self.isMutiProcessInit:
            self.mutiProcessInit()
            self.isMutiProcessInit = True

        # 向上取整 train_data_num / self.read_process_num
        each_reader_process_train_num = (self.train_data_num + self.read_process_num -
                                         self.train_data_num % self.read_process_num) / self.read_process_num

        for i in range(self.read_process_num):
            p_train_data = train_data[i * each_reader_process_train_num: min(self.train_data_num, (i + 1) * each_reader_process_train_num)]
            p_train_labels = train_labels[i * each_reader_process_train_num: min(self.train_data_num, (i + 1) * each_reader_process_train_num)]
            self.ReaderInstances.append(DataReader(self.batch_size, self.PS, self.buffer_queue, self.templates, self.vec_length, p_train_data, p_train_labels, self.scale))
        for i in range(self.read_process_num):
            self.ReaderInstances[i].start()

    def mutiProcessInit(self):

        #初始化用0填充的vec_array和bias_net的value
        templates = {}
        template_vec_value = copy.deepcopy(self.vec_array)
        if self.bias:
            template_bias_net = self.bias
            templates['bias_net_value'] = template_bias_net.value
            templates['bias_net_map_of_slot'] = template_bias_net.map_of_slot
        else:
            templates['bias_net_value'] = None
            templates['bias_net_map_of_slot'] = None
        templates['vec_array'] = template_vec_value
        self.templates = templates

    def readInstance(self):
        buffer_obj = self.buffer_queue.get()
        self.vec_array = buffer_obj['vec_array']
        if self.bias:
            bias_net_value = buffer_obj['bias_net_value']
            self.bias.value = bias_net_value
        self.slot_to_fids = buffer_obj['slot_to_fids']
        labels = buffer_obj['labels']
        return labels

    def update(self):
        '''
        TODO apply gradient to optimizer
        '''
        for i in range(self.batch_size):
            print self.vec_array
            if self.bias:
                for slot_id in self.slot_to_fids[i]:
                    for fid in self.slot_to_fids[i][slot_id]:
                        self.PS[fid] += self.bias.getValue(slot_id, i) * 100
            for slot_id in self.vec_array:
                for fid in self.slot_to_fids[i][slot_id]:
                    v_fid = fid + (1 << 62)
                    self.PS[v_fid] += self.vec_array[slot_id][i, :] * 100
                    '''
                    if slot_id == 1:
                        print self.PS[v_fid]
                    '''

class DataReader(Process):
    def __init__(self, batch_size, PS, queue, templates, vec_length, train_data, train_labels=None, scale=0.01):
        Process.__init__(self)
        self.batch_size = batch_size
        self.PS = PS
        self.queue = queue
        self.templates = templates
        self.vec_length = vec_length
        self.train_data = train_data
        self.train_labels = train_labels
        self.scale = scale

    def run(self):
        print 'batch', self.batch_size
        train_data_offset = 0
        train_data_num = len(self.train_data)
        train_steps = train_data_num / self.batch_size
        if (train_data_num % self.batch_size) != 0:
            train_steps += 1
        for i in range(train_steps):
            step_offset = min(train_data_num - train_data_offset,
                              self.batch_size)
            if self.train_labels:
                labels = self.train_labels[train_data_offset: train_data_offset + step_offset]
                labels.extend([0] * (self.batch_size - step_offset))
                labels = np.array(labels, ndmin=2).T
            vec_array = self.templates['vec_array']
            bias_net_value = self.templates['bias_net_value']
            for slot in vec_array:
                vec_array[slot].fill(0.0)
            if bias_net_value:
                bias_net_value.fill(0.0)
            bias_net_map_of_slot = self.templates['bias_net_map_of_slot']
            slot_to_fids = {i: collections.defaultdict(list) for i in range(self.batch_size)}
            for j in range(step_offset):
                instance = self.train_data[train_data_offset]
                for fid in instance:
                    slot_id = (fid >> 52) & 1023
                    v_fid = fid + (1 << 62)
                    if v_fid not in self.PS and slot_id in self.vec_length:
                        self.PS[v_fid] = np.random.normal(0, self.scale, (self.vec_length[slot_id]))
                    if fid not in self.PS:
                        self.PS[fid] = random.normalvariate(0, self.scale)
                    if slot_id in self.vec_length:
                        vec_array[slot_id][j, :] += self.PS[v_fid]
                    if bias_net_value:
                        bias_net_value[j: bias_net_map_of_slot[slot_id]] += self.PS[fid]
                    slot_to_fids[j][slot_id].append(fid)
                train_data_offset += 1
            # print train_data_offset, step_offset

            buffer_obj = {}
            buffer_obj['vec_array'] = vec_array
            buffer_obj['bias_net_value'] = bias_net_value
            buffer_obj['slot_to_fids'] = slot_to_fids
            if self.train_labels:
                buffer_obj['labels'] = labels
            else:
                buffer_obj['labels'] = None
            # print self.queue.empty(), self.queue.full()
            self.queue.put(buffer_obj)


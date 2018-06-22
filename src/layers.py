# encoding=utf-8

from utils import extractInputNode
from ioManger import Vec
from utils import extractSlotId

import numpy as np
import collections


class Layer(object):
    def __init__(self, inputs, outputs, **kwargs):
        self.inbound_nodes = []
        self.outbound_nodes = []
        if not isinstance(inputs, list):
            inputs = [inputs]
        self.inputs = inputs
        if not isinstance(outputs, list):
            outputs = [outputs]
        self.outputs = outputs
        if 'scale' in kwargs:
            self.scale = kwargs.get('scale')
        self.config = kwargs
        self.type = 'Baselayer'
        self.input_size = None
        self.output_size = None
        self.trainable = kwargs.get('trainable', True)
        self.initInputLayer = False
        self.isInputLayer = False
        self.name = 'Default'
        self.debug = kwargs.get('debug', False)
        self.debug = False

    def forward(self, mode='train'):
        if self.debug:
            print 'Forwarding... ', self
        return self._forward(mode)

    def _forward(self, mode='train'):
        raise NotImplementedError

    def backward(self):
        if self.debug:
            print 'Backwarding... ', self
        self.backprob_gradient = sum(node.gradient for node in self.outbound_nodes)
        return self._backward()

    def _backward(self):
        raise NotImplementedError

    def setup(self):
        '''
        1. need to get output_size and input_size if is input layer
        2. need to comlete first initlization
        '''
        self._check()
        self._setup()

    def _setup(self):
        raise NotImplementedError

    def _check(self):
        pass

    def update(self, updateDelta):
        # TODO 全局setup时检查shape
        # print self, 'weight:', self.weight.shape, 'update', updateDelta.shape
        self.weight += updateDelta

    def transmitForward(self, value):
        '''
        transmit forward value to outbound_nodes
        '''
        for node in self.outbound_nodes:
            node.receiveForwardValue(value)

    def transmitBackward(self, gradient):
        '''
        transmit backward value to inbound_nodes
        if gradient is a Sequence like \'list\', the gradient will trainsmit
            to each inbound node
        '''
        if isinstance(gradient, collections.Sequence):
            for i, g in enumerate(gradient):
                self.inbound_nodes[i].receiveBackwardValue(g)
        else:
            for node in self.inbound_nodes:
                node.receiveBackwardValue(gradient)

    def setIoManger(self, ioManger):
        self.ioManger = ioManger

    def initInput(self):
        '''
        this step executes before function of setup
        need to return inbound_nodes of this layer
        '''
        return extractInputNode(self.inputs, self.ioManger)

    def __repr__(self):
        inputs_str = ','.join([node.name for node in self.inbound_nodes])
        outputs_str = ','.join([node.name for node in self.outbound_nodes])
        return '[LAYER]: ' + self.name + ',Input: ' + inputs_str + ',Output: ' + outputs_str

    def setName(self, layer_id):
        self.name = self.type + ':' + str(layer_id)


class FakeLayer(Layer):
    def __init__(self, inputs, outputs, **kwargs):
        super(MergeLayer, self).__init__(inputs, outputs, **kwargs)
        self.type = 'Fake'

    def _setup(self):
        self.value = np.zeros(self.input_size)
        self.output_size = self.input_size


class MergeLayer(Layer):
    def __init__(self, inputs, outputs, **kwargs):
        super(MergeLayer, self).__init__(inputs, outputs, **kwargs)
        self.type = 'Merge'

    def _setup(self):
        self._inputStartIndex = {}
        self._inputEndIndex = {}
        tot_vec_length = 0
        for index, node in enumerate(self.inbound_nodes):
            bathchsize, vec_length = node.output_size
            self._inputStartIndex[index] = tot_vec_length
            tot_vec_length += vec_length
            self._inputEndIndex[index] = tot_vec_length
        self.value = np.zeros((bathchsize, tot_vec_length))
        self.gradient = np.zeros((bathchsize, tot_vec_length))
        self.output_size = (bathchsize, tot_vec_length)

    def _forward(self, model='train'):
        for index, node in enumerate(self.inbound_nodes):
            self.value[:, self._inputStartIndex[index]: self._inputEndIndex[index]] = node.value
        return self.value

    def _backward(self):
        for index, node in enumerate(self.inbound_nodes):
            self.gradient[:, self._inputStartIndex[index]: self._inputEndIndex[index]] = node.value
        return self.gradient


class DotLayer(Layer):
    def __init__(self, inputs, outputs, **kwargs):
        super(DotLayer, self).__init__(inputs, outputs, **kwargs)
        self.type = 'Dot'
        self.hiddenNum = kwargs.get('hiddenNum', 10)

    def _setup(self):
        if len(self.inbound_nodes) > 1:
            raise Exception('Dotlayer only allow one input')
        if (not isinstance(self.hiddenNum, int)) or self.hiddenNum < 1:
            raise Exception('Dotlayer\'s hiddenNum must more than 1')
        batchsize, vec_length = self.inbound_nodes[0].output_size
        self.weight = np.random.normal(0.0, self.scale, (vec_length, self.hiddenNum))
        self.gradient = np.zeros((batchsize, self.hiddenNum))
        self.output_size = (batchsize, self.hiddenNum)

    def _forward(self, mode='train'):
        '''
        将中间计算结果放在_value中，用于后期计算梯度
        TODO 封装self.inbound_nodes[0], 对用户友好
        '''
        self._input_value = self.inbound_nodes[0].value
        self._value = self._input_value.dot(self.weight)
        return self._value

    def _backward(self):
        self.gradient = self._input_value.T.dot(self.backprob_gradient)
        return self.backprob_gradient.dot(self.weight.T)


class SigmoidLayer(Layer):
    def __init__(self, inputs, outputs, **kwargs):
        super(SigmoidLayer, self).__init__(inputs, outputs, **kwargs)

    def _setup(self):
        if len(self.inbound_nodes) > 1:
            raise Exception('SigmoidLayer only allow one input')
        self.output_size = self.inbound_nodes[0].output_size

    def _sigmoiFunc(self, value):
        value = np.clip(value, -10, 10)
        return 1.0 / (1 + np.exp(-value))

    def _forward(self, model='train'):
        self._input_value = self.inbound_nodes[0].value
        self._value = self._sigmoiFunc(self._input_value)
        return self._value

    def _backward(self):
        return self.backprob_gradient * self._value * (1 - self._value)


class FMGroupLayer(Layer):
    def __init__(self, inputs, outputs, **kwargs):
        '''
        Args:
            inputs: here it must be \'Input\'
            outputs: list of output's node's name
            fm_config:[fm_group:(row:(slot_1, slot_4, ...),
                col:(slot_3, slot_5...),
                vec_length: n), (), ()...]]
                ex: [[('slot_1', 'slot_2'), ('slot_5'), 8)], [('slot_4'), ('slot_8'), 8)]]
        '''
        super(FMGroupLayer, self).__init__(inputs, outputs, **kwargs)
        self.fm_config = kwargs.get('fm_config')
        self.inputLayer = 0
        self.type = 'FmGroup'
        if inputs != ['Input']:
            raise Exception("FMGroupLayer's inputs only can be [\'Input\']")

    def initInput(self):
        nodes = []
        for fm_group in self.fm_config:
            row_slots = fm_group[0]
            col_slots = fm_group[1]
            vec_length = fm_group[2]
            row_vec_caches = []
            col_vec_caches = []

            for r_slot in row_slots:
                r_id = extractSlotId(r_slot)
                row_vec_caches.append(Vec(r_id, self.ioManger))
            for c_slot in col_slots:
                c_id = extractSlotId(c_slot)
                col_vec_caches.append(Vec(c_id, self.ioManger))
            for r_vec in row_vec_caches:
                for c_vec in col_vec_caches:
                    blob1, blob2 = r_vec.cross(c_vec, vec_length)
                    nodes.append(blob1)
                    nodes.append(blob2)

        self.inputLayer = 1
        return nodes

    def _setup(self):
        if self.inputLayer:
            batchsize, _ = self.inbound_nodes[0].output_size
            self.batchsize = batchsize
            self.inner_config = []
            for i in range(len(self.inbound_nodes) / 2):
                self.inner_config.append((self.inbound_nodes[i * 2],
                                          self.inbound_nodes[i * 2 + 1]))
            self.output_size = (batchsize, len(self.inner_config))
        else:
            pass

    def _forward(self, model='train'):
        if self.inputLayer:
            self._value = []
            for group in self.inner_config:
                r = np.sum(group[0].value * group[1].value, axis=1, keepdims=True)
                self._value.append(r)
            self._value = np.concatenate(self._value, axis=1)
            return self._value
        else:
            pass

    def _backward(self):
        if self.inputLayer:
            gradients = []
            for i in range(len(self.inner_config)):
                g = self.backprob_gradient[:, i].reshape(self.batchsize, 1)
                gradients.append(g * self.inbound_nodes[i * 2 + 1].value)
                gradients.append(g * self.inbound_nodes[i * 2].value)
            return gradients
        else:
            pass


class SumPoolingLayer(Layer):
    def __init__(self, inputs, outputs, **kwargs):
        super(SumPoolingLayer, self).__init__(inputs, outputs, **kwargs)
        self.type = 'SumPooling'

    def _setup(self):
        self.output_size = self.inbound_nodes[0].output_size

    def _forward(self, mode='train'):
        self._value = sum(x.value for x in self.inbound_nodes)
        return self._value

    def _backward(self):
        return self.backprob_gradient

    def _check(self):
        input_shapes = [x.shape for x in self.inbound_nodes]
        shape = input_shapes[0]
        for each in input_shapes[1:]:
            if each != shape:
                raise Exception('SumPoolingLayer\'s inputs only allow one kind of size')


class RowPoolingLayer(Layer):
    def __init__(self, inputs, outputs, **kwargs):
        super(RowPoolingLayer, self).__init__(inputs, outputs, **kwargs)
        self.type = 'RowPooling'

    def _setup(self):
        batchsize, _ = self.inbound_nodes[0].output_size
        self.output_size = (batchsize, 1)
        self.input_size = self.inbound_nodes[0].output_size

    def _forward(self, mode='train'):
        self._input_value = self.inbound_nodes[0].value
        self._value = np.sum(self._input_value, axis=1, keepdims=True)
        return self._value

    def _backward(self):
        g = np.ones(self.input_size)
        return self.backprob_gradient * g

    def _check(self):
        if len(self.inputs) > 1:
            raise Exception('RowPoolingLayer only allow one input')

class Node():
    def __init__(self, name):
        self.name = name
        self.type = 'middle'
        self.inbound_layers = []
        self.outbound_layers = []
        self.candidate_values = []
        self.candidate_gradints = []
        self.input_size = None
        self.output_size = None
        self.finished_nodes = 0
        self.needPrepare = 1

    @property
    def value(self):
        if self.needPrepare:
            self.prepare()
        return self._value

    @property
    def gradient(self):
        if self.needPrepare:
            self.prepare()
        return self._gradient

    def prepare(self):
        self.needPrepare = 0
        if self.candidate_gradints and self.candidate_values:
            raise Exception('update value and gradients at the same time!')
        if self.candidate_values:
            self._value = sum(self.candidate_values)
            self.candidate_values = []
        if self.candidate_gradints:
            self._gradient = sum(self.candidate_gradints)
            self.candidate_gradints = []

    def setup(self):
        self.input_size = self.inbound_layers[0].output_size
        for layer in self.inbound_layers[1:]:
            if layer.output_size != self.input_size:
                raise Exception('Node has over one type of input')
        self.output_size = self.input_size

    def receiveForwardValue(self, value):
        self.needPrepare = 1
        self.candidate_values.append(value)

    def receiveBackwardValue(self, gradient):
        self.needPrepare = 1
        self.candidate_gradints.append(gradient)

    def clear(self):
        self.candidate_values = []
        self.candidate_gradints = []

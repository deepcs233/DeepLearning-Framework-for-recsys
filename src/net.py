import Queue
from layers import Node


class Net():
    def __init__(self, ioManger, scale=0.01):
        '''
        scale only work when layer not set scale
        '''
        self.layers = []
        self.nodes = {}
        self.in_decreasing_depth = []
        self.in_increasing_depth = []
        self.layers_in_decreasing_depth = []
        self.nodes_no_include_input = []# used for clear node
        self.finallyOutputNode = None
        self.ioManger = ioManger
        self.layer_count = 0
        self.scale = scale

    def getInputStatus(self, layer):
        layer.isInputLayer = False
        for each in layer.inputs:
            if 'slot' in each or 'bias' in each or 'Input' in each:
                layer.isInputLayer = True
                break

    def add(self, layer):
        layer.setIoManger(self.ioManger)
        layer.setName(self.layer_count)
        if not hasattr(layer, 'scale'):
            layer.scale = self.scale
        self.layer_count += 1
        self.layers.append(layer)
        self.getInputStatus(layer)
        if layer.isInputLayer:
            if len(layer.inputs) != 1:
                raise Exception('Only allow to include node of type \'Input\' if this layer includes type \'Input\'')
            inputNodes = layer.initInput()
            for input_node in inputNodes:
                self.nodes[input_node.name] = input_node
                layer.inbound_nodes.append(input_node)
                input_node.outbound_layers.append(layer)
        else:
            for input_name in layer.inputs:
                if input_name in self.nodes:
                    new_node = self.nodes[input_name]
                else:
                    new_node = Node(input_name)
                    self.nodes[input_name] = new_node
                new_node.outbound_layers.append(layer)
                self.nodes_no_include_input.append(new_node)
                layer.inbound_nodes.append(new_node)

        if 'Output' in layer.outputs:
            if len(layer.outputs) != 1:
                raise Exception('Only allow to include type \'Output\' if this layer includes node of type \'Ouput\'')
            if self.finallyOutputNode:
                raise Exception('Only allow one Output node')
            new_node = Node('Output')
            self.finallyOutputNode = new_node
            self.nodes['Output'] = new_node
            new_node.inbound_layers.append(layer)
            layer.outbound_nodes.append(new_node)
            self.nodes_no_include_input.append(new_node)
        else:
            for output_name in layer.outputs:
                if output_name in self.nodes:
                    new_node = self.nodes[output_name]
                else:
                    new_node = Node(output_name)
                    self.nodes[output_name] = new_node
                new_node.inbound_layers.append(layer)
                layer.outbound_nodes.append(new_node)
                self.nodes_no_include_input.append(new_node)

    def setup(self):
        '''
        init net after add all layers
        complete all prepartion for next compution
        '''
        layers_queue = Queue.Queue()
        layers_queue.put(self.finallyOutputNode.inbound_layers[0])
        visited_layers = set([])
        while not layers_queue.empty():
            layer = layers_queue.get()
            self.in_decreasing_depth.append(layer)
            visited_layers.add(layer)
            for node in layer.inbound_nodes:
                if node.type != 'input':
                    for t_layer in node.inbound_layers:
                        if t_layer not in visited_layers:
                            visited_layers.add(t_layer)
                            layers_queue.put(t_layer)
        self.in_increasing_depth = self.in_decreasing_depth[::-1]
        for layer in self.layers:
            if layer not in visited_layers:
                raise Exception('existing \'useless\' layer')
        for layer in self.in_increasing_depth:
            if not layer.isInputLayer:
                input_size = layer.inbound_nodes[0].output_size
                for node in layer.inbound_nodes[1:]:
                    if node.output_size != input_size:
                        raise Exception('layer has over one type of input')
                layer.input_size = input_size
                layer.setup()
            else:
                ''' only input layer needs to use ioManger to setup '''
                layer.setup()
            for node in layer.outbound_nodes:
                node.finished_nodes += 1
                if node.finished_nodes == len(node.inbound_layers):
                    node.setup()

        self.ioManger.initArray(self.scale)
        # build gradient update list
        self.needUpdateObjList = []
        for layer in self.layers:
            if hasattr(layer, 'gradient'):
                self.needUpdateObjList.append(layer)
            if layer.isInputLayer:
                for node in layer.inbound_nodes:
                    self.needUpdateObjList.append(node)

    def setLossFunc(self, lossFunc):
        self.lossFunc = lossFunc

    def forward(self, mode='train'):
        '''
        complete all layers' forward operation
        '''
        for layer in self.in_increasing_depth:
            value = layer.forward(mode)
            layer.transmitForward(value)
        preds = self.finallyOutputNode.value
        if mode == 'test':
            self.clear()
             # print preds.shape, type(preds)
        self._preds = preds
        return preds

    def computeLossAndGradient(self, y_labels):
        '''
        complete loss and gradient of output node
        '''
        loss = self.lossFunc(self._preds, y_labels)
        gradient = self.lossFunc.gradient(self._preds, y_labels)
        self.finallyOutputNode.receiveBackwardValue(gradient)
        return loss

    def backward(self):
        '''
        complete all layers' backward operation
        '''
        for layer in self.in_decreasing_depth:
            # print 'backwarding... ', layer
            gradient = layer.backward()
            layer.transmitBackward(gradient)

    def push(self):
        '''
        complete parameters update
        '''
        gradient_list = []
        for Obj in self.needUpdateObjList:
            gradient_list.append(Obj.gradient)
        update_list = self.optimizer.get_updates(gradient_list)
        for index, Obj in enumerate(self.needUpdateObjList):
            Obj.update(update_list[index])
        self.ioManger.update()

    def clear(self):
        '''
        clear nodes cache to avoid memory leak
        '''
        for node in self.nodes_no_include_input:
            node.clear()

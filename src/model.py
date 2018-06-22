# encoding=utf-8
import ConfigParser
import random
import math
import sys
import time

import numpy as np

#from utils import memoryCheck
from ioManger import ioManger
from optimizer import SGD
from conf import default_config_path


class Model():
    def __init__(self, net, optimizer, lossFunc, ioManger, PS=None,
                 metricsFunc=None, config_path=None):
        '''
        metricsFunc must have `attr` name
        '''
        self.net = net
        self.optimizer = optimizer
        self.lossFunc = lossFunc
        self.ioManger = ioManger
        self.random_seed = random.randint(0,100000)
        cf = ConfigParser.ConfigParser()
        if config_path:
            self.config = cf.read(config_path)
        else:
            self.config = cf.read(default_config_path)
        if PS is None:
            self.PS = {}
        else:
            self.PS = PS
        if metricsFunc is None:
            self.metricsFunc = []
        else:
            self.metricsFunc = metricsFunc

    def build(self):
        print('Net setuping...')
        self.net.setup()
        self.net.setLossFunc(self.lossFunc)
        self.net.optimizer = self.optimizer
        self.metrics_name = []
        for metricsFunc in self.metricsFunc:
            self.metrics_name.append(metricsFunc.name)
        # self.metric_funcs = []
        # self.loss_func = None

    def fit(self, X, y, verbose=True, epochs=10, shuffle=True,
            initial_epoch=0, print_every_steps=None, validation_split=0.,
            val_x=None, val_y=None):
        '''
        traintest_per_steps must bigger than batch_size
        print_every_steps: every some steps, logger will print the loss
        TODO 滚动进度条
        '''
        print('Start training...')
        data_num = len(X)
        train_num = int(data_num * (1 - validation_split))
        if print_every_steps is None:
            print_every_steps = train_num / 10
        if not (val_x and val_y):
            val_x = X[train_num:]
            val_y = y[train_num:]

        train_data = X[: train_num]
        train_label = y[: train_num]


        for now_epoch in range(initial_epoch, epochs):
            if shuffle:
                random.seed(self.random_seed)
                random.shuffle(train_data)
                random.seed(self.random_seed)
                random.shuffle(train_label)
            self.ioManger.sentData(train_data, train_label)
            print_every_steps_i = 0
            history = {'label': [], 'pred': [], 'loss': []}

            epoch_start_time = time.time()
            fetch_time_cost = 0.
            forward_time_cost = 0.
            compute_loss_time_cost = 0.
            backward_time_cost = 0.
            push_time_cost = 0.

            train_steps = train_num / self.ioManger.batch_size
            if train_num % self.ioManger.batch_size > 0:
                train_steps += 1

            for i in range(train_steps):
                before_fetch_time = time.time()
                y_labels = self.ioManger.readInstance()
                after_fetch_time = time.time()
                y_preds = self.net.forward()
                after_forward_time = time.time()
                losses = self.net.computeLossAndGradient(y_labels)
                after_computer_loss_time = time.time()
                self.net.backward()
                after_backward_time = time.time()
                self.net.push()
                after_push_time = time.time()

                fetch_time_cost += after_fetch_time - before_fetch_time
                forward_time_cost += after_forward_time - after_fetch_time
                compute_loss_time_cost += after_computer_loss_time - after_forward_time
                backward_time_cost += after_backward_time - after_computer_loss_time
                push_time_cost += after_push_time - after_backward_time

                history['label'].extend(y_labels)
                history['pred'].extend(y_preds)
                history['loss'].append(losses)
                print_every_steps_i += self.ioManger.batch_size
                if print_every_steps_i > print_every_steps:
                    print_every_steps_i -= print_every_steps
                    avg_loss = sum(history['loss']) / len(history['loss'])
                    avg_pred = np.sum(history['pred']) / len(history['pred'])
                    avg_label = float(sum(history['label'])) / len(history['label'])
                    metrics_score = self.getMetricsScore(history['pred'], history['label'])
                    print('epoch: %s, training complete %s, logloss: %.6s, avgpred: %.6s, avglabel: %.6s' % (now_epoch, self.ioManger.batch_size * (i + 1), avg_loss, avg_pred, avg_label))
                    ''' need print avg loss, and metrics'''
                    history = {'label': [], 'pred': [], 'loss': []}
            '''
            after epoch training , now validating
            '''
            '''
            TODO need do metric verbose
            '''
            epoch_end_time = time.time()
            epoch_time_cost = epoch_end_time - epoch_start_time
            print('each epoch time cost:%.6ss, every sec can train %.6s instance' % (epoch_time_cost, train_num / epoch_time_cost))
            print('Every 100: fetch time: %.6ss, forward time: %.6ss, '
                  'compute loss time: %.6ss, backward time: %.6ss, '
                  'push time: %.6ss' % (fetch_time_cost * 100 / train_num,
                                        forward_time_cost * 100 / train_num,
                                        compute_loss_time_cost * 100 / train_num,
                                        backward_time_cost * 100 / train_num,
                                        push_time_cost * 100 / train_num))
            if validation_split > 0:
                val_preds = self.predict(val_x)
                avg_loss = self.lossFunc(val_preds, val_y)
                avg_pred = np.sum(val_preds) / len(val_preds)
                avg_label = float(sum(val_y)) / len(val_y)
                # metrics_score = self.getMetricsScore(val_preds, val_y)
                print("epoch: %s, validation complete, avgloss: %.6s, avgpred: %.6s, avglabel: %.6s" % (now_epoch, avg_loss, avg_pred, avg_label))

    def getMetricsScore(self, preds, labels):
        metrics_score = {}
        for metricsFunc in self.metricsFunc:
            score = metricsFunc(preds, labels)
            metrics_score[metricsFunc.name] = score
        return metrics_score

    def predict(self, vali_data):
        '''
        TODO: now each step only predict one instance
        '''
        vali_preds = []
        self.ioManger.sentData(vali_data)
        predict_steps = len(vali_data) / self.ioManger.batch_size
        if len(vali_data) % self.ioManger.batch_size > 0:
            predict_steps += 1
        for i in range(predict_steps):
            self.ioManger.readInstance()
            y_preds = self.net.forward('test')
            vali_preds.extend(y_preds)
        vali_preds = np.concatenate(vali_preds, axis=0)
        vali_preds = vali_preds[: len(vali_data)]
        return vali_preds

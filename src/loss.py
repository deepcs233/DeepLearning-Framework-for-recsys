import random
import sys

import numpy as np

# from utils import memoryCheck
epsilon = 1e-7


class mean_squared_error():
    def __init__(self):
        self.name = 'mean_squared_error'

    def __call__(self, y_pred, y_true):
        return np.linalg.norm(np.array(y_pred) - np.array(y_true))

    def gradient(self, y_pred, y_true):
        return np.matrix(np.mean(np.array(y_pred) - np.array(y_true)))


class mean_absolute_error():
    def __init__(self):
        self.name = 'mean_absolute_error'

    def __call__(self, y_pred, y_true):
        return np.mean(np.abs(np.array(y_pred) - np.array(y_true)))

    def gradient(self, y_pred, y_true):
        raise Exception('mean_absolute_error has not gradient!')


class cross_entropy_error():
    def __init__(self):
        self.name = 'cross_entropy_error'

    def __call__(self, y_pred, y_true):
        t = np.array(y_true)
        p = np.array(y_pred)
        # print y_true, y_pred
        avgloss = -np.mean(t * np.log(np.clip(p, epsilon, None)) + (1 - t) * np.log(np.clip(1 - p, epsilon, None)))

        return avgloss

    def gradient(self, y_pred, y_true):
        loss = (y_true - y_pred) / (y_pred * (1 - y_pred))
        return loss


class hinge_error():
    def __init__(self):
        self.name = 'hinge_error'

    def __call__(self, y_pred, y_true):
        return np.mean(np.clip(1 - np.array(y_pred) * np.array(y_true), 0, None))

    def gradient(self, y_pred, y_true):
        pass

class kullback_leibler_divergence():
    def __init__(self):
        self.name = 'kld'

    def __call__(self, y_pred, y_true):
        y_pred = np.clip(y_pred, epsilon, 1.)
        y_true = np.clip(y_true, epsilon, 1.)
        return np.sum(y_true * np.log(y_true / y_pred), axis=-1)

    def gradient(self, y_pred, y_true):
        pass


mse = MSE = mean_squared_error()
mae = MAE = mean_absolute_error()
cee = CEE = cross_entropy_error()
he = HE = hinge_error()
kld = KLD = kullback_leibler_divergence()

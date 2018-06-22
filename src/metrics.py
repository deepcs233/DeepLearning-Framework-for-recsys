import numpy as np

from loss import mean_squared_error
from loss import mean_absolute_error
from loss import cross_entropy_error
from loss import hinge_error
from loss import kullback_leibler_divergence

epsilon = 1e-7


class binary_accuracy():
    def __init__(self):
        self.name = 'binary_accuracy'

    def __call__(self, y_pred, y_true):
        return np.mean(np.equal(y_true, np.round(y_pred)), axis=-1)


mse = MSE = mean_squared_error()
mae = MAE = mean_absolute_error()
cee = CEE = cross_entropy_error()
he = HE = hinge_error()
kld = KLD = kullback_leibler_divergence()

binary_accuracy = binary_accuracy()

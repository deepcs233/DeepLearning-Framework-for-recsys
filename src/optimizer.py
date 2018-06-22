#encoding=utf-8
import numpy as np

class Optimizer(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'clipnorm', 'clipvalue'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to optimizer: ' + str(k))
        self.__dict__.update(kwargs)

    def clip_norm(self, gradient):
        if None in grads:
            raise ValueError('An operation has `None` for gradient. '
                             'Please make sure that all of your ops have a '
                             'gradient defined (i.e. are differentiable). ')
        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = np.linalg.norm(gradient)
            grads = [np.clip(g, self.clipnorm, norm) for g in gradient]
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        return grads
#TODO !1111111111111


class SGD(Optimizer):
    '''
    momentum: float >= 0. Parameter that accelerates SGD
        in the relevant direction and dampens oscillations.
    decay: float >= 0. Learning rate decay over each update.
    nesterov: boolean. Whether to apply Nesterov momentum.
    '''
    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, iterations=0,  **kwards):
        super(SGD, self).__init__(**kwards)
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.nesterov = nesterov
        self.iterations = iterations
        self.firstRun = 1

    def get_updates(self, grads):
        '''
        每调用一次该方法, epoch数+1
        TODO 不应该每调用一次epoch数加一
        '''
        updates = []

        if self.firstRun:
            self.firstRun = 0
            self.momentum_cache = []
            for grad in grads:
                self.momentum_cache.append(np.zeros_like(grad))

        for i in range(len(grads)):
            self.momentum_cache[i] = self.momentum * self.momentum_cache[i] + self.lr * grads[i]
            if self.nesterov:
                updates.append(self.momentum * self.momentum_cache[i] - self.lr * grads[i])
            else:
                updates.append(self.momentum_cache[i])

        if self.decay > 0:
            self.lr *= (1. / (1. + self.decay * self.iterations))

        self.iterations += 1
        return updates

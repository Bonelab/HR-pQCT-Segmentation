"""
Written by Nathan Neeteson
A class to encapsulate the logic for calculation of learning rate and momentum
using linear interpolation.
"""

import numpy as np


# it is useful to encapsulate the logic this way because in the future we may
# want to use some other method for scheduling learning rate and momentum
# and in that case the training script only needs to have a different object
# instantiated for scheduling, with the same interface
class OptimizerSchedulerLinear(object):

    def __init__(self, epochs, lrs, moms):
        # the three non-self inputs should be lists of epoch checkpoints
        # and what the learning rate and momentum should be at each.
        # the object will then linearly interpolate to set the parameters in
        # the optimizer for a given epoch
        self.epochs = epochs
        self.lrs = lrs
        self.moms = moms

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_lr(self):
        return np.interp(self.epoch, self.epochs, self.lrs)

    def get_mom(self):
        return np.interp(self.epoch, self.epochs, self.moms)


class OptimizerSchedulerLogarithmicLR(object):

    def __init__(self, epochs, lrs, moms):
        # the three non-self inputs should be lists of epoch checkpoints
        # and what the learning rate and momentum should be at each.
        # the object will then calculate the power of ten for the
        self.epochs = epochs
        self.lr_orders = np.log10(lrs)
        self.moms = moms

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_lr(self):
        lr_order = np.interp(self.epoch, self.epochs, self.lr_orders)
        return 10 ** lr_order

    def get_mom(self):
        return np.interp(self.epoch, self.epochs, self.moms)

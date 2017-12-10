from __future__ import print_function
__author__ = 'jianglu'
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class Dropout(object):
    # dropput layer
    def __init__(self, input, p=0, layer_index=0):
        # input is a tensor
        self.layername = 'Dropout' + str(layer_index)
        self.input = input
        self.p = p
        self.retain_prob = 1 - self.p
        self.srng = RandomStreams()
        self.mask = self.srng.binomial(self.input.shape, p=self.retain_prob, dtype=theano.config.floatX)

        if self.p > 0:
            self.output = (self.input * self.mask) / self.retain_prob
        else:
            self.output = self.input



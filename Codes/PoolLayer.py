from theano.tensor.signal.pool import pool_2d

__author__='lujiang'

class PoolLayer(object):

    def __init__(self, input, poolsize = (2,2), layer_index=0):

        self.layername = 'Pool' + str(layer_index)

        self.input = input

        self.output = pool_2d(
            input = input,
            ds = poolsize,
            ignore_border = False,
            mode='max'
        )
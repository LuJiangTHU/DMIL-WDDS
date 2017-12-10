from __future__ import print_function
__author__ = 'jianglu'
import theano.tensor as T

class LRN(object):

    def __init__(self, input, alpha=1e-4, k=2, beta=0.75, n=5, layer_index=0):

        # input is a 4d tensor:(batchsize,channels,raws,cols)
        self.layername = 'LRN' + str(layer_index)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        self.input = input
        if n % 2 == 0:
            raise NotImplementedError("only works with odd")

        half_n = self.n // 2
        input_sqr = T.sqr(input)
        b,ch,r,c = T.shape(input)
        extra_channels = T.alloc(0, b,ch+2*half_n,r,c)
        input_sqr = T.set_subtensor(extra_channels[:,half_n:half_n+ch,:,:],input_sqr)

        scale = self.k
        for i in range(self.n):
            scale += self.alpha * input_sqr[:,i:i+ch,:,:]
        scale = scale ** self.beta

        self.output = self.input / scale


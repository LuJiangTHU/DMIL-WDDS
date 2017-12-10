__author__='lujiang'
import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import relu


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, layer_index=0):

        self.layername = 'Full' + str(layer_index)
        self.input = input

        if W is None:
            W_bound = numpy.sqrt(6./(n_in+n_out))
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(low= -W_bound, high= W_bound, size=(n_in, n_out)),
                    dtype= theano.config.floatX
                    ),
                    borrow =True
                    )
        else:
            self.W = theano.shared(
                value = W.astype(theano.config.floatX),
                borrow = True )
        self.W.name = self.layername + '#W'


        if b is None:
            self.b = theano.shared(
                numpy.zeros((n_out,),dtype= theano.config.floatX),
                    borrow =True
                    )
        else:
            self.b = theano.shared(
                value = b.astype(theano.config.floatX),
                borrow = True
                    )
        self.b.name = self.layername + '#b'


        self.output = relu(T.dot(input, self.W) + self.b)

        # parameters of the model
        self.params = [self.W, self.b]
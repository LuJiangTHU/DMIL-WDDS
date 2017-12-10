import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import relu
from theano.tensor.nnet.nnet import softmax
import lasagne

__author__='lujiang'

class SoftmaxLayer(object):

    def __init__(self, rng, input, n_in, n_out, W = None, b = None, layer_index=0):

        self.layername = 'Softmax' + str(layer_index)
        self.input = input

        # W
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
                borrow = True
                    )
        self.W.name = self.layername + '#W'

        # b
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

        output = relu(T.dot(input, self.W) + self.b)

        self.softmax_output = softmax(output)

        self.pred = self.softmax_output.argmax(axis=1)

        # store parameters of this layer
        self.params = [self.W, self.b]



    # y:1-7
    def cross_entropy(self, y):
        return T.mean(lasagne.objectives.categorical_crossentropy(self.softmax_output,y))

    def squared_error(self, y):
       return T.mean(0.5*T.sum((self.softmax_output-y)*(self.softmax_output-y),axis=1))


    # error
    def errors(self,y):
        # (batchsize,)
        y_label = T.argmax(y,axis=1)
        # check if y_label has same dimension of pred
        if y_label.ndim != self.pred.ndim:
            raise TypeError(
                'y should have the same shape as self.pred',
                ('y',y_label.type,'pred',self.pred.type)
            )

        # check if y is of the correct data type
        if y_label.dtype.startswith('int'):
            return T.mean(T.neq(self.pred,y_label))
        else:
            raise NotImplementedError()



        
      
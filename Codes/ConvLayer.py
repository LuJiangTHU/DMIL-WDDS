import numpy
import theano
from theano.tensor.nnet import conv2d,relu

__author__='lujiang'

class ConvLayer(object):

    def __init__(self, rng, input, filter_shape, W = None, b = None, stride=(1,1), layer_index=0):

        self.layername = 'Conv' + str(layer_index)
        self.input = input
        # num input feature maps * filter height * filter width
        fan_in = numpy.prod(filter_shape[1:])

        # num output feature maps * filter height * filter width
        fan_out = (filter_shape[0]*numpy.prod(filter_shape[2:]))

        # W
        if W is None:
            W_bound = numpy.sqrt(6./(fan_in+fan_out))
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(low= -W_bound, high= W_bound, size=filter_shape),
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
                numpy.zeros((filter_shape[0],),dtype= theano.config.floatX),
                    borrow =True
                    )
        else:
            self.b = theano.shared(
                value = b.astype(theano.config.floatX),
                borrow = True
                    )
        self.b.name = self.layername + '#b'

        # batch size * num feature maps * feature map height * width
        conv_out = conv2d(
            input= input,
            filters = self.W,
            filter_shape = filter_shape,
            border_mode='half',
            subsample=stride
        )

        self.output = relu(conv_out + self.b.dimshuffle('x',0,'x','x'))

        # prepared for the last 3 FC layers
        valid_conv_out = conv2d(
            input = input,
            filters = self.W,
            filter_shape = filter_shape,
            subsample=stride
        )
        self.nopad_output = relu(valid_conv_out + self.b.dimshuffle('x',0,'x','x'))

        # store parameters of this layer
        self.params = [self.W, self.b]
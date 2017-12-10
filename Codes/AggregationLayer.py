import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d,softmax,sigmoid,relu

__author__='lujiang'

class AggregationLayer(object):

    def __init__(self, rng, input, filter_shape, W = None, b = None, agg_func='soft', layer_index=0, alpha = 2.5):

        self.layername = 'Aggregation' + str(layer_index)
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
        # size : batch size * kinds num * height * width
        self.valid_conv_out = conv2d(
            input = input,
            filters = self.W,
            filter_shape = filter_shape,
        )


        if agg_func == 'soft':
            # ------- combining function is softmax ,ML logistic regression (multi-class)--------
            # batchsize * kind num * height * width
            self.pro_instance = sigmoid(self.valid_conv_out + self.b.dimshuffle('x',0,'x','x'))
            # batchsize * kind num *(height*width)
            self.pro_spread = self.pro_instance.flatten(ndim=3)

            # (batchsize,kind num)
            self.pro_bag = (T.exp(alpha*self.pro_spread) * self.pro_spread).sum(axis=2) / \
                        T.exp(alpha*self.pro_spread).sum(axis=2)

            # (batchsize,)
            self.pred = self.pro_bag.argmax(axis=1)

            # multi-image to one plant object
            self.ensemblePred = self.pro_bag.mean(axis=0).argmax()
            # -------------------------------------------------------------------------
        elif agg_func == 'max':
            # ------- combining function is hardmax ,ML logistic regression (multi-class)--------
            # batchsize * kind num * height * width
            self.pro_instance = sigmoid(self.valid_conv_out + self.b.dimshuffle('x',0,'x','x'))
            # batchsize * kind num *(height*width)
            self.pro_spread = self.pro_instance.flatten(ndim=3)

            # (batchsize,kind num)
            self.pro_bag = self.pro_spread.max(axis=2)

            # (batchsize,)
            self.pred = self.pro_bag.argmax(axis=1)

            # multi-image to one plant object
            self.ensemblePred = self.pro_bag.mean(axis=0).argmax()
            # -------------------------------------------------------------------------
        else:

            # ------- combining function is mean, ML logistic regression (multi-class)--------
            # batchsize * kind num * height * width
            self.pro_instance = sigmoid(self.valid_conv_out + self.b.dimshuffle('x',0,'x','x'))
            # batchsize * kind num *(height*width)
            self.pro_spread = self.pro_instance.flatten(ndim=3)

            # (batchsize,kind num)
            self.pro_bag = self.pro_spread.mean(axis=2)

            # (batchsize,)
            self.pred = self.pro_bag.argmax(axis=1)

            # multi-image to one plant object
            self.ensemblePred = self.pro_bag.mean(axis=0).argmax()
            # -------------------------------------------------------------------------

        # store parameters of this layer
        self.params = [self.W, self.b]


    def squared_error(self, y):
       return T.mean(0.5*T.sum((self.pro_bag-y)*(self.pro_bag-y),axis=1))


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



        
      
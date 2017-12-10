from __future__ import print_function
__author__ = 'jianglu'
from ConvLayer import ConvLayer
from LRN import LRN
from PoolLayer import PoolLayer
from HiddenLayer import HiddenLayer
from SoftmaxLayer import SoftmaxLayer
import theano
import theano.tensor as T
import lasagne

class VGG_S_CNN(object):

    def __init__(self,rng,weight,use_last_layer_weight=False):
        # weight: list type,0-31,the first 32 W and b
        x = T.tensor4('x')
        y = T.imatrix('y')
        learning_rate = T.scalar('learning_rate')
        self.layer1_input = x  #224

        self.layer1 = ConvLayer(rng,input=self.layer1_input, filter_shape=(96,3,7,7), layer_index=1,
                                stride=(2,2),W=weight[0], b=weight[1])  #output:
        self.LRN_1 = LRN(input = self.layer1.nopad_output, layer_index=1)

        self.pool1  = PoolLayer(input = self.LRN_1.output, poolsize=(3,3), layer_index=1)  # 138

        self.layer2 = ConvLayer(rng,input=self.pool1.output,filter_shape=(256,96,5,5),layer_index=2,
                                W=weight[2], b=weight[3])  #output:
        self.pool2  = PoolLayer(input = self.layer2.nopad_output, poolsize=(2,2), layer_index=2) #67

        self.layer3 = ConvLayer(rng,input=self.pool2.output, filter_shape=(512,256,3,3),layer_index=3,
                                W=weight[4], b=weight[5]) #
        self.layer4 = ConvLayer(rng,input=self.layer3.output, filter_shape=(512,512,3,3),layer_index=4,
                                W=weight[6], b=weight[7]) #
        self.layer5 = ConvLayer(rng,input=self.layer4.output, filter_shape=(512,512,3,3),layer_index=5,
                                W=weight[8], b=weight[9]) #
        self.pool3  = PoolLayer(input = self.layer5.output, poolsize=(3,3), layer_index=3) #23


        if use_last_layer_weight:  #use weight of layer14/15/16
            self.layer6 = HiddenLayer(rng,input=self.pool3.output.flatten(ndim=2),n_in=18432,n_out=1024,layer_index=6,
                                        W=weight[10],b=weight[11])
            self.layer7 = HiddenLayer(rng,input=self.layer6.output,n_in=1024,n_out=1024,layer_index=7,
                                        W=weight[12],b=weight[13])
            self.layer8 = SoftmaxLayer(rng,input=self.layer7.output,n_in=1024,n_out=7,layer_index=8,
                                        W=weight[14],b=weight[15])
        else:
            self.layer6 = HiddenLayer(rng,input=self.pool3.output.flatten(ndim=2),n_in=18432,n_out=1024,layer_index=6)
            self.layer7 = HiddenLayer(rng,input=self.layer6.output,n_in=1024,n_out=1024,layer_index=7)
            self.layer8 = SoftmaxLayer(rng,input=self.layer7.output,n_in=1024,n_out=7,layer_index=8)

        # the objective loss
        self.loss = self.layer8.squared_error(y)

        self.params = self.layer8.params +self.layer7.params+self.layer6.params +self.layer5.params\
                      +self.layer4.params +self.layer3.params +self.layer2.params +self.layer1.params

        # optimization methods
        updates = lasagne.updates.nesterov_momentum(self.loss,self.params,
                                                    learning_rate=learning_rate,
                                                    momentum=0.9)
        #updates=lasagne.updates.rmsprop(self.loss, self.params, learning_rate, rho=0.9, epsilon=1e-06)


        self.train = theano.function(inputs = [x,y,learning_rate],
                                     outputs= [self.loss,self.layer8.errors(y)],
                                     updates= updates,
                                     allow_input_downcast=True)

        self.test = theano.function(inputs = [x,y],
                                    outputs= [self.loss,self.layer8.errors(y)],
                                    allow_input_downcast=True)

        #self.detection = theano.function(
        #            inputs = [x,y],
        #            outputs= [self.layer16.pred,self.layer16.errors(y),self.layer16.pro_instance,self.layer16.pro_bag],
        #            allow_input_downcast=True )


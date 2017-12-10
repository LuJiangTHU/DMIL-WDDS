from __future__ import print_function
__author__ = 'jianglu'
from ConvLayer import ConvLayer
from LRN import LRN
from PoolLayer import PoolLayer
from AggregationLayer import AggregationLayer
import theano
import theano.tensor as T
import lasagne

class VGG_S_FCN(object):

    def __init__(self,rng,weight,agg_func='soft',use_last_layer_weight=False):
        # weight: list type,0-15,the first 16 W and b
        x = T.tensor4('x')
        y = T.imatrix('y')
        learning_rate = T.scalar('learning_rate')
        self.layer1_input = x  #832

        self.layer1 = ConvLayer(rng,input=self.layer1_input, filter_shape=(96,3,7,7), layer_index=1,
                                stride=(2,2),W=weight[0], b=weight[1])  #output:413
        self.LRN_1 = LRN(input = self.layer1.nopad_output, layer_index=1)

        self.pool1  = PoolLayer(input = self.LRN_1.output, poolsize=(3,3), layer_index=1)  # 138

        self.layer2 = ConvLayer(rng,input=self.pool1.output,filter_shape=(256,96,5,5),layer_index=2,
                                W=weight[2], b=weight[3])  #output: 134
        self.pool2  = PoolLayer(input = self.layer2.nopad_output, poolsize=(2,2), layer_index=2) #67

        self.layer3 = ConvLayer(rng,input=self.pool2.output, filter_shape=(512,256,3,3),layer_index=3,
                                W=weight[4], b=weight[5]) #67
        self.layer4 = ConvLayer(rng,input=self.layer3.output, filter_shape=(512,512,3,3),layer_index=4,
                                W=weight[6], b=weight[7]) #67
        self.layer5 = ConvLayer(rng,input=self.layer4.output, filter_shape=(512,512,3,3),layer_index=5,
                                W=weight[8], b=weight[9]) #67
        self.pool3  = PoolLayer(input = self.layer5.output, poolsize=(3,3), layer_index=3) #23

        ## change FC1/FC2/FC3 layers to Convlayers14/15/16

        if use_last_layer_weight:  #use weight of layer14/15/16
            self.layer6 = ConvLayer(rng,input=self.pool3.output,filter_shape=(1024,512,6,6),layer_index=6,
                                     W=weight[10],b=weight[11])  #18
            self.layer7 = ConvLayer(rng,input=self.layer6.nopad_output,filter_shape=(1024,1024,1,1),layer_index=7,
                                     W=weight[12],b=weight[13])   #18
            self.layer8 = AggregationLayer(rng,input=self.layer7.nopad_output,filter_shape=(7,1024,1,1),layer_index=8,
                                            W=weight[14],b=weight[15],agg_func=agg_func) #18
        else:
            self.layer6 = ConvLayer(rng,input=self.pool3.output,filter_shape=(1024,512,6,6),layer_index=6)#18
            self.layer7 = ConvLayer(rng,input=self.layer6.nopad_output,filter_shape=(1024,1024,1,1),layer_index=7)#18
            self.layer8 = AggregationLayer(rng,input=self.layer7.nopad_output,filter_shape=(7,1024,1,1),layer_index=8,
                                           agg_func=agg_func)#18


        # the objective loss
        self.loss = self.layer8.squared_error(y)

        self.params = self.layer8.params+self.layer7.params+self.layer6.params+self.layer5.params\
                     +self.layer4.params+self.layer3.params+self.layer2.params +self.layer1.params

        # optimization methods
        updates = lasagne.updates.nesterov_momentum(self.loss,self.params,
                                                    learning_rate=learning_rate,
                                                    momentum=0.9)

        self.train = theano.function(inputs = [x,y,learning_rate],
                                     outputs= [self.loss,self.layer8.errors(y)],
                                     updates= updates,
                                     allow_input_downcast=True)

        self.test = theano.function(inputs = [x,y],
                                    outputs= [self.loss,self.layer8.errors(y)],
                                    allow_input_downcast=True)

        self.detection = theano.function(
                   inputs = [x,y],
                   outputs= [self.layer8.pred,self.layer8.errors(y),
                             self.layer8.pro_instance,self.layer8.pro_bag],
                   allow_input_downcast=True )


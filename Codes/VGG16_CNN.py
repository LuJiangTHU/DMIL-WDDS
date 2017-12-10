from __future__ import print_function
__author__ = 'jianglu'
from ConvLayer import ConvLayer
from PoolLayer import PoolLayer
from HiddenLayer import HiddenLayer
from SoftmaxLayer import SoftmaxLayer
import theano
import theano.tensor as T
import lasagne

class VGG16_CNN(object):

    def __init__(self,rng,weight,use_last_layer_weight=False):
        # weight: list type,0-31,the first 32 W and b
        x = T.tensor4('x')
        y = T.imatrix('y')
        learning_rate = T.scalar('learning_rate')
        self.layer1_input = x  #224

        self.layer1 = ConvLayer(rng,input=self.layer1_input, filter_shape=(64,3,3,3), layer_index=1, W=weight[0], b=weight[1])
        self.layer2 = ConvLayer(rng,input=self.layer1.output,filter_shape=(64,64,3,3),layer_index=2, W=weight[2], b=weight[3])
        self.pool1  = PoolLayer(input = self.layer2.output, poolsize=(2,2), layer_index=1) #112

        self.layer3 = ConvLayer(rng,input=self.pool1.output, filter_shape=(128,64,3,3),layer_index=3, W=weight[4], b=weight[5])
        self.layer4 = ConvLayer(rng,input=self.layer3.output,filter_shape=(128,128,3,3),layer_index=4,W=weight[6], b=weight[7])
        self.pool2  = PoolLayer(input = self.layer4.output, poolsize=(2,2), layer_index=2) #56

        self.layer5 = ConvLayer(rng,input=self.pool2.output, filter_shape=(256,128,3,3),layer_index=5,W=weight[8], b=weight[9])
        self.layer6 = ConvLayer(rng,input=self.layer5.output,filter_shape=(256,256,3,3),layer_index=6,W=weight[10], b=weight[11])
        self.layer7 = ConvLayer(rng,input=self.layer6.output,filter_shape=(256,256,3,3),layer_index=7,W=weight[12], b=weight[13])
        self.pool3  = PoolLayer(input = self.layer7.output, poolsize=(2,2), layer_index=3) #28

        self.layer8 = ConvLayer(rng,input=self.pool3.output, filter_shape=(512,256,3,3),layer_index=8,W=weight[14], b=weight[15])
        self.layer9 = ConvLayer(rng,input=self.layer8.output,filter_shape=(512,512,3,3),layer_index=9,W=weight[16], b=weight[17])
        self.layer10= ConvLayer(rng,input=self.layer9.output,filter_shape=(512,512,3,3),layer_index=10,W=weight[18],b=weight[19])
        self.pool4  = PoolLayer(input = self.layer10.output, poolsize=(2,2), layer_index=4) #14

        self.layer11 = ConvLayer(rng,input=self.pool4.output,filter_shape=(512,512,3,3),layer_index=11,W=weight[20],b=weight[21])
        self.layer12 = ConvLayer(rng,input=self.layer11.output,filter_shape=(512,512,3,3),layer_index=12,W=weight[22],b=weight[23])
        self.layer13 = ConvLayer(rng,input=self.layer12.output,filter_shape=(512,512,3,3),layer_index=13,W=weight[24],b=weight[25])
        self.pool5   = PoolLayer(input = self.layer13.output, poolsize=(2,2), layer_index=5) #7

        if use_last_layer_weight:  #use weight of layer14/15/16
            self.layer14 = HiddenLayer(rng,input=self.pool5.output.flatten(ndim=2),n_in=25088,n_out=1024,layer_index=14,
                                        W=weight[26],b=weight[27])
            self.layer15 = HiddenLayer(rng,input=self.layer14.output,n_in=1024,n_out=1024,layer_index=15,
                                        W=weight[28],b=weight[29])
            self.layer16 = SoftmaxLayer(rng,input=self.layer15.output,n_in=1024,n_out=7,layer_index=16,
                                        W=weight[30],b=weight[31])
        else:
            self.layer14 = HiddenLayer(rng,input=self.pool5.output.flatten(ndim=2),n_in=25088,n_out=1024,layer_index=14)
            self.layer15 = HiddenLayer(rng,input=self.layer14.output,n_in=1024,n_out=1024,layer_index=15)
            self.layer16 = SoftmaxLayer(rng,input=self.layer15.output,n_in=1024,n_out=7,layer_index=16)


        # the objective loss
        self.loss = self.layer16.squared_error(y)

        self.params = self.layer16.params+self.layer15.params+self.layer14.params+self.layer13.params+self.layer12.params\
                     +self.layer11.params+self.layer10.params+self.layer9.params +self.layer8.params +self.layer7.params\
                     +self.layer6.params +self.layer5.params +self.layer4.params +self.layer3.params +self.layer2.params\
                     +self.layer1.params

        # optimization methods
        updates = lasagne.updates.nesterov_momentum(self.loss,self.params,
                                                    learning_rate=learning_rate,
                                                    momentum=0.9)

        self.train = theano.function(inputs = [x,y,learning_rate],
                                     outputs= [self.loss,self.layer16.errors(y)],
                                     updates= updates,
                                     allow_input_downcast=True)

        self.test = theano.function(inputs = [x,y],
                                    outputs= [self.loss,self.layer16.errors(y)],
                                    allow_input_downcast=True)

        #self.detection = theano.function(
        #            inputs = [x,y],
        #            outputs= [self.layer16.pred,self.layer16.errors(y),self.layer16.pro_instance,self.layer16.pro_bag],
        #            allow_input_downcast=True )


from __future__ import print_function
__author__ = 'jianglu'
import os
import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt
from VGG16_FCN import VGG16_FCN

## different random seeds
rng = np.random.RandomState(23455)
#rng = np.random.RandomState(1123)
#rng = np.random.RandomState(564)
#rng = np.random.RandomState(9852)

##############################
#Loading pre-training params #
##############################
print('...loading pre-training params')
f = open('/home/trunk/disk1/lujiang/vgg16.pkl')
VGG16model = pickle.load(f)
f.close()
W = VGG16model['param values'][0:26]
meanValue = VGG16model['mean value']


#######################
##### Build Model #####
#######################
print('...building the model')
VGG16_FCN_model = VGG16_FCN(rng,weight=W, use_last_layer_weight=False)


#######################
##### Train Model #####
#######################
print('......training')

batchsize = 2
dirs_Trn = []
dirs_Trn.extend(os.listdir('/home/trunk/disk1/lujiang/WDD/fold1'))
dirs_Trn.extend(os.listdir('/home/trunk/disk1/lujiang/WDD/fold2'))
dirs_Trn.extend(os.listdir('/home/trunk/disk1/lujiang/WDD/fold3'))
dirs_Trn.extend(os.listdir('/home/trunk/disk1/lujiang/WDD/fold4'))

dirs_Tst = os.listdir('/home/trunk/disk1/lujiang/WDD/fold5')

f1 = open('/home/trunk/disk1/lujiang/WDD/fold1_label.txt')
f2 = open('/home/trunk/disk1/lujiang/WDD/fold2_label.txt')
f3 = open('/home/trunk/disk1/lujiang/WDD/fold3_label.txt')
f4 = open('/home/trunk/disk1/lujiang/WDD/fold4_label.txt')
list1 = np.asarray([x.split() for x in f1.readlines()])
list2 = np.asarray([x.split() for x in f2.readlines()])
list3 = np.asarray([x.split() for x in f3.readlines()])
list4 = np.asarray([x.split() for x in f4.readlines()])
Trn_label_list = np.concatenate((list1,list2,list3,list4))
f1.close()
f2.close()
f3.close()
f4.close()

f = open('/home/trunk/disk1/lujiang/WDD/fold5_label.txt')
Tst_label_list = np.asarray([x.split() for x in f.readlines()])
f.close()

fold_batch = len(dirs_Tst) / batchsize

n_epoch = 20
base_lr = 0.00005
best_tst_error = np.inf
current_tst_loss = np.inf
best_iter = 0

for epoch in range(n_epoch):
    # learning rate for different epoch
    lr = 5** -(epoch/5) * base_lr
    print('*********** learning_rate: %f ************' % lr)

    for k in range(fold_batch*4):
        iter = fold_batch*4 * epoch + k +1
        # load train data
        batch_input_x = []
        batch_input_y = []

        if k < fold_batch:  #fold-1
            for i in range(batchsize):
                index = k*batchsize + i
                img = plt.imread('/home/trunk/disk1/lujiang/WDD/fold1/'+dirs_Trn[index]).transpose(2,0,1)\
                      -meanValue.reshape(-1,1,1)
                label = np.zeros(7,dtype='int32')
                hot_id = int(Trn_label_list[np.where(Trn_label_list==dirs_Trn[index])[0],1][0])
                label[hot_id-1] = label[hot_id-1]+1
                batch_input_x.append(img)
                batch_input_y.append(label)

        elif k < fold_batch*2:  #fold-2
            for i in range(batchsize):
                index = k*batchsize + i
                img = plt.imread('/home/trunk/disk1/lujiang/WDD/fold2/'+dirs_Trn[index]).transpose(2,0,1)\
                      -meanValue.reshape(-1,1,1)
                label = np.zeros(7,dtype='int32')
                hot_id = int(Trn_label_list[np.where(Trn_label_list==dirs_Trn[index])[0],1][0])
                label[hot_id-1] = label[hot_id-1]+1
                batch_input_x.append(img)
                batch_input_y.append(label)

        elif k < fold_batch*3:  #fold-3
            for i in range(batchsize):
                index = k*batchsize + i
                img = plt.imread('/home/trunk/disk1/lujiang/WDD/fold3/'+dirs_Trn[index]).transpose(2,0,1)\
                      -meanValue.reshape(-1,1,1)
                label = np.zeros(7,dtype='int32')
                hot_id = int(Trn_label_list[np.where(Trn_label_list==dirs_Trn[index])[0],1][0])
                label[hot_id-1] = label[hot_id-1]+1
                batch_input_x.append(img)
                batch_input_y.append(label)

        else :                #fold-4
            for i in range(batchsize):
                index = k*batchsize + i
                img = plt.imread('/home/trunk/disk1/lujiang/WDD/fold4/'+dirs_Trn[index]).transpose(2,0,1)\
                      -meanValue.reshape(-1,1,1)
                label = np.zeros(7,dtype='int32')
                hot_id = int(Trn_label_list[np.where(Trn_label_list==dirs_Trn[index])[0],1][0])
                label[hot_id-1] = label[hot_id-1]+1
                batch_input_x.append(img)
                batch_input_y.append(label)

        batch_input_x = np.asarray(batch_input_x)
        batch_input_y = np.asarray(batch_input_y,dtype='int32')
        loss_iter, error_iter = VGG16_FCN_model.train(batch_input_x,batch_input_y,lr)
        print('Epoch:%d, Iter:%d, TrainMinibatch: %i/%i, miniLoss:%f, miniTrainError: %f%%' %
              (epoch+1, iter, k+1, fold_batch*4, loss_iter, error_iter*100.))

        if iter % 1000 ==0:
            tst_loss =[]
            tst_error = []
            for k in range(fold_batch):
                # load tst data
                batch_input_x = []
                batch_input_y = []
                for i in range(batchsize):
                    index = k*batchsize + i
                    img = plt.imread('/home/trunk/disk1/lujiang/WDD/fold5/'+dirs_Tst[index]).transpose(2,0,1)\
                          -meanValue.reshape(-1,1,1)
                    label = np.zeros(7,dtype='int32')
                    hot_id = int(Tst_label_list[np.where(Tst_label_list==(dirs_Tst[index]))[0],1][0])
                    label[hot_id-1] = label[hot_id-1]+1
                    batch_input_x.append(img)
                    batch_input_y.append(label)

                batch_input_x = np.asarray(batch_input_x)
                batch_input_y = np.asarray(batch_input_y,dtype='int32')

                loss_iter, error_iter = VGG16_FCN_model.test(batch_input_x,batch_input_y)
                tst_loss.append(loss_iter)
                tst_error.append(error_iter)
                print('TestSet(Epoch:%d, Iter:%d) minibatch: %i/%i, miniTestError: %f%%' %
                      (epoch+1, iter, k+1, fold_batch, error_iter*100.))



            tst_loss = np.mean(np.asarray(tst_loss))
            tst_error = np.mean(np.asarray(tst_error))

            print('-------------------------------------------------------------------------------------')
            print('Epoch:%d, Iter:%d-- TestError:%f%%, TestLoss: %f' %
                                (epoch+1, iter, tst_error*100.,  tst_loss))
            print('-------------------------------------------------------------------------------------')


            # save the best model
            if tst_error < best_tst_error:
                best_tst_error = tst_error
                best_iter = iter
                current_tst_loss = tst_loss


                model_file = 'VGG16_Mean_tstfold5_1/'+str(epoch+1)+'_'+str(iter)+'.h5'
                f = h5py.File(model_file)
                for p in VGG16_FCN_model.params:
                    f[p.name] = p.get_value()
                f.close()


print('Optimation complete.')
print('Best test score of %f%% obtained at iteration %i, with test loss %f' %
      (best_tst_error*100., best_iter,current_tst_loss))

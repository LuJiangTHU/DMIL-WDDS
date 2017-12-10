from __future__ import print_function
__author__ = 'jianglu'
import os
import numpy as np
import h5py
import skimage.measure
import matplotlib.patches as mpatches
import pickle
import matplotlib.pyplot as plt

import matplotlib
import cv2
from VGG16_FCN import VGG16_FCN
from VGG_S_FCN import VGG_S_FCN

rng = np.random.RandomState(23455)

##############################
#Loading pre-training params #
##############################
print('...loading pre-training params')
f = open('/home/trunk/disk1/lujiang/vgg16.pkl')
VGG16model = pickle.load(f)
meanValue = VGG16model['mean value']
f.close()

classname = ['Powdery Mildew','Smut','Black Chaff', 'Stripe Rust', 'Leaf Blotch', 'Leaf Rust','Healthy']
show_factor =0.8            # clip the border arround
area_threshold = 20000      # restrain the small area
binary_threshold= 0.9       # binary image threshold

f = h5py.File('VGG16_FCN_tstfold5_4/12_43000.h5')
W1 = f['Conv1#W'][:]
b1 = f['Conv1#b'][:]
W2 = f['Conv2#W'][:]
b2 = f['Conv2#b'][:]
W3 = f['Conv3#W'][:]
b3 = f['Conv3#b'][:]
W4 = f['Conv4#W'][:]
b4 = f['Conv4#b'][:]
W5 = f['Conv5#W'][:]
b5 = f['Conv5#b'][:]
W6 = f['Conv6#W'][:]
b6 = f['Conv6#b'][:]
W7 = f['Conv7#W'][:]
b7 = f['Conv7#b'][:]
W8 = f['Conv8#W'][:]
b8 = f['Conv8#b'][:]
W9 = f['Conv9#W'][:]
b9 = f['Conv9#b'][:]
W10 = f['Conv10#W'][:]
b10 = f['Conv10#b'][:]
W11 = f['Conv11#W'][:]
b11 = f['Conv11#b'][:]
W12 = f['Conv12#W'][:]
b12 = f['Conv12#b'][:]
W13 = f['Conv13#W'][:]
b13 = f['Conv13#b'][:]
W14 = f['Conv14#W'][:]
b14 = f['Conv14#b'][:]
W15 = f['Conv15#W'][:]
b15 = f['Conv15#b'][:]
W16 = f['Aggregation16#W'][:]
b16 = f['Aggregation16#b'][:]
f.close()

W_soft = [W1,b1,W2,b2,W3,b3,W4,b4,W5,b5,W6,b6,W7,b7,W8,b8,W9,b9,W10,b10,W11,
     b11,W12,b12,W13,b13,W14,b14,W15,b15,W16,b16]


f = h5py.File('VGG16_Hard_tstfold5_1/8_29000.h5')
W1 = f['Conv1#W'][:]
b1 = f['Conv1#b'][:]
W2 = f['Conv2#W'][:]
b2 = f['Conv2#b'][:]
W3 = f['Conv3#W'][:]
b3 = f['Conv3#b'][:]
W4 = f['Conv4#W'][:]
b4 = f['Conv4#b'][:]
W5 = f['Conv5#W'][:]
b5 = f['Conv5#b'][:]
W6 = f['Conv6#W'][:]
b6 = f['Conv6#b'][:]
W7 = f['Conv7#W'][:]
b7 = f['Conv7#b'][:]
W8 = f['Conv8#W'][:]
b8 = f['Conv8#b'][:]
W9 = f['Conv9#W'][:]
b9 = f['Conv9#b'][:]
W10 = f['Conv10#W'][:]
b10 = f['Conv10#b'][:]
W11 = f['Conv11#W'][:]
b11 = f['Conv11#b'][:]
W12 = f['Conv12#W'][:]
b12 = f['Conv12#b'][:]
W13 = f['Conv13#W'][:]
b13 = f['Conv13#b'][:]
W14 = f['Conv14#W'][:]
b14 = f['Conv14#b'][:]
W15 = f['Conv15#W'][:]
b15 = f['Conv15#b'][:]
W16 = f['Aggregation16#W'][:]
b16 = f['Aggregation16#b'][:]
f.close()

W_max = [W1,b1,W2,b2,W3,b3,W4,b4,W5,b5,W6,b6,W7,b7,W8,b8,W9,b9,W10,b10,W11,
     b11,W12,b12,W13,b13,W14,b14,W15,b15,W16,b16]

f = h5py.File('VGG16_Mean_tstfold5_1/10_36000.h5')
W1 = f['Conv1#W'][:]
b1 = f['Conv1#b'][:]
W2 = f['Conv2#W'][:]
b2 = f['Conv2#b'][:]
W3 = f['Conv3#W'][:]
b3 = f['Conv3#b'][:]
W4 = f['Conv4#W'][:]
b4 = f['Conv4#b'][:]
W5 = f['Conv5#W'][:]
b5 = f['Conv5#b'][:]
W6 = f['Conv6#W'][:]
b6 = f['Conv6#b'][:]
W7 = f['Conv7#W'][:]
b7 = f['Conv7#b'][:]
W8 = f['Conv8#W'][:]
b8 = f['Conv8#b'][:]
W9 = f['Conv9#W'][:]
b9 = f['Conv9#b'][:]
W10 = f['Conv10#W'][:]
b10 = f['Conv10#b'][:]
W11 = f['Conv11#W'][:]
b11 = f['Conv11#b'][:]
W12 = f['Conv12#W'][:]
b12 = f['Conv12#b'][:]
W13 = f['Conv13#W'][:]
b13 = f['Conv13#b'][:]
W14 = f['Conv14#W'][:]
b14 = f['Conv14#b'][:]
W15 = f['Conv15#W'][:]
b15 = f['Conv15#b'][:]
W16 = f['Aggregation16#W'][:]
b16 = f['Aggregation16#b'][:]
f.close()

W_avg = [W1,b1,W2,b2,W3,b3,W4,b4,W5,b5,W6,b6,W7,b7,W8,b8,W9,b9,W10,b10,W11,
     b11,W12,b12,W13,b13,W14,b14,W15,b15,W16,b16]


#######################
##### Build Model #####
#######################
print('...building the model')
model_soft = VGG16_FCN(rng,weight=W_soft, use_last_layer_weight=True,agg_func='soft')
model_max =  VGG16_FCN(rng,weight=W_max, use_last_layer_weight=True,agg_func='max')
model_avg =  VGG16_FCN(rng,weight=W_avg, use_last_layer_weight=True,agg_func='avg')


#######################
##### Test Model #####
#######################
print('......test')

# try random sequence display
batchsize = 1
dirs_test = os.listdir('WDD/fold5/')

# read label
f = open('WDD/fold5_label.txt')
tst_label_list = np.asarray([x.split() for x in f.readlines()])
f.close()

# score_size = (832-224)/32+1


test_error_soft = []
test_error_max = []
test_error_avg = []

soft_error_list = open('soft_error_list.txt','a')
max_error_list = open('max_error_list.txt','a')
avg_error_list = open('avg_error_list.txt','a')
for index in range(len(dirs_test)):
    print(index+1)
    # load tst data
    x = []
    y = []
    img_org = plt.imread('WDD/fold5/'+dirs_test[index])
    img = img_org.transpose(2,0,1)-meanValue.reshape(-1,1,1)
    label = np.zeros(7,dtype='int32')
    hot_id = int(tst_label_list[np.where(tst_label_list==(dirs_test[index]))[0],1][0])
    label[hot_id-1] = label[hot_id-1]+1

    batch_input_x = np.asarray(x.append(img))
    batch_input_y = np.asarray(y.append(label))

    prediction_soft, tst_error_soft, pro_soft, pro_bag_soft = model_soft.detection(x,y)
    prediction_max, tst_error_max, pro_max, pro_bag_max = model_max.detection(x, y)
    prediction_avg, tst_error_avg, pro_avg, pro_bag_avg = model_avg.detection(x, y)

    if tst_error_soft>0:
        soft_error_list.write(dirs_test[index]+"\n")
    if tst_error_max>0:
        max_error_list.write(dirs_test[index]+"\n")
    if tst_error_avg>0:
        avg_error_list.write(dirs_test[index]+"\n")

    test_error_soft.append(tst_error_soft)
    test_error_max.append(tst_error_max)
    test_error_avg.append(tst_error_avg)
    print('tst_img/'+dirs_test[index]+'---test number: %i,Soft Error: %f%%' %( index+1, tst_error_soft*100.))
    print('tst_img/' + dirs_test[index] + '---test number: %i,Max Error: %f%%' % (index + 1, tst_error_max * 100.))
    print('tst_img/' + dirs_test[index] + '---test number: %i,Avg Error: %f%%' % (index + 1, tst_error_avg * 100.))

    fig = plt.figure()

    plt.imshow(img_org)
    plt.axis('off')

    prediction_map_soft = cv2.resize(pro_soft[:,prediction_soft,:,:].squeeze(),(832,832),interpolation=cv2.INTER_CUBIC)
    re,prediction_map_BW_soft = cv2.threshold(prediction_map_soft,binary_threshold,1,cv2.THRESH_BINARY)
    prediction_map_BW_soft = np.asarray(prediction_map_BW_soft,dtype=np.uint8)
    _, contours, _ = cv2.findContours(prediction_map_BW_soft,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        box = cv2.boundingRect(contours[i])
        box_area = box[2]*box[3]
        if box_area > area_threshold:
            w_bias = int(box[2]*(1-show_factor)*0.5)
            h_bias = int(box[3]*(1-show_factor)*0.5)
            w = int(box[2]*show_factor)
            h = int(box[3]*show_factor)
            rect = mpatches.Rectangle((box[0]+w_bias,box[1]+h_bias),w,h,fill=False,edgecolor='springgreen',linewidth=2)
            plt.gca().add_patch(rect)
            plt.text(box[0]+70,box[1]+70,classname[prediction_soft[0]], size = 6,
                         family = "Times New Roman", color = "black", style = "italic",
                         bbox = dict(facecolor = "springgreen", alpha = 1))

    img_soft = img_org[:, :, 0] * prediction_map_soft
    plt.imsave('WDD_save/'+dirs_test[index]+'_soft_'+classname[prediction_soft[0]]+'.eps',img_soft)
    # ax[1].imshow(img_soft)



    # box approximation by opencv findContours: max
    prediction_map_max = cv2.resize(pro_max[:,prediction_max,:,:].squeeze(),(832,832),interpolation=cv2.INTER_CUBIC)
    re, prediction_map_BW_max = cv2.threshold(prediction_map_max,binary_threshold,1,cv2.THRESH_BINARY)
    prediction_map_BW_max = np.asarray(prediction_map_BW_max,dtype=np.uint8)
    _, contours, _ = cv2.findContours(prediction_map_BW_max,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        box = cv2.boundingRect(contours[i])
        box_area = box[2]*box[3]
        if box_area > area_threshold/3:
            w_bias = int(box[2]*(1-0.9)*0.5)
            h_bias = int(box[3]*(1-0.9)*0.5)
            w = int(box[2]*show_factor)
            h = int(box[3]*show_factor)
            rect = mpatches.Rectangle((box[0]+w_bias,box[1]+h_bias),w,h,fill=False,edgecolor='pink',linewidth=2)
            plt.gca().add_patch(rect)
            plt.text(box[0]+30,box[1]+20,classname[prediction_max[0]], size = 6,
                         family = "Times New Roman", color = "black", style = "italic",
                         bbox = dict(facecolor = "pink", alpha = 1))
    img_max = img_org[:, :, 0] * prediction_map_max
    plt.imsave('WDD_save/'+dirs_test[index]+'_max_'+classname[prediction_max[0]]+'.eps',img_max)
    # ax[2].imshow(img_max)



    # box approximation by opencv findContours: avg
    prediction_map_avg = cv2.resize(pro_avg[:,prediction_avg,:,:].squeeze(),(832,832),interpolation=cv2.INTER_CUBIC)
    re, prediction_map_BW_avg = cv2.threshold(prediction_map_avg,binary_threshold,1,cv2.THRESH_BINARY)
    prediction_map_BW_avg = np.asarray(prediction_map_BW_avg,dtype=np.uint8)
    _, contours, _ = cv2.findContours(prediction_map_BW_avg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        box = cv2.boundingRect(contours[i])
        box_area = box[2]*box[3]
        if box_area > area_threshold:
            w_bias = int(box[2]*(1-show_factor)*0.5)
            h_bias = int(box[3]*(1-show_factor)*0.5)
            w = int(box[2]*show_factor)
            h = int(box[3]*show_factor)
            rect = mpatches.Rectangle((box[0]+w_bias,box[1]+h_bias),w,h,fill=False,edgecolor='lightblue',linewidth=2)
            plt.gca().add_patch(rect)
            plt.text(box[0]+70,box[1]+70,classname[prediction_avg[0]], size = 6,
                         family = "Times New Roman", color = "black", style = "italic",
                         bbox = dict(facecolor = "lightblue", alpha = 1))
    img_avg = img_org[:, :, 0] * prediction_map_avg
    plt.imsave('WDD_save/'+dirs_test[index]+'_avg_'+classname[prediction_avg[0]]+'.eps',img_avg)
    fig.savefig('WDD_save/'+dirs_test[index]+'.eps',bbox_inches='tight', pad_inches=0.0, format='eps')
    # ax[3].imshow(img_avg)
    # plt.show()


test_error_soft = np.mean(np.asarray(test_error_soft))
test_error_max = np.mean(np.asarray(test_error_max))
test_error_avg = np.mean(np.asarray(test_error_avg))
soft_error_list.close()
max_error_list.close()
avg_error_list.close()
print('Test complete.')
print('Total test error is %f%%,%f%%,%f%%' % (test_error_soft*100.,test_error_max*100.,test_error_avg*100.))
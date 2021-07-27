# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 13:05:58 2020

@author: deepu
"""

import glob
import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Permute,Layer,Conv3D,ConvLSTM2D,Conv3DTranspose,Dense,Flatten,Reshape,Dropout,MaxPooling3D,UpSampling3D,add,Input,Activation
from tensorflow.keras.models import Sequential,Model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import math


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


import os
#import skvideo.io
#from skimage.transform import resize
#from skimage.io import imsave

video_root_path = 'D:\HAR DATASETS av vio etc\HAR Dataset\CV\Fold 1' 
size = (226, 226)

class MemoryUnit(tf.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = tf.Variable(tf.random.normal([self.mem_dim, self.fea_dim], stddev=0.35),trainable=True,shape=tf.TensorShape([self.mem_dim, self.fea_dim]))  # M x C
        self.bias = None
        self.shrink_thres= shrink_thres
        # self.hard_sparse_shrink_opt = nn.Hardshrink(lambd=shrink_thres)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.shape[1])###check this
        
        self.weight=tf.random.uniform(shape=self.weight.shape, minval=-stdv, maxval=stdv)

        if self.bias is not None:
            self.bias=tf.random.uniform(shape=self.bias.shape, minval=-stdv, maxval=stdv)

    def __call__(self, input):
        print(self.weight.shape,"weight")
        print(input.shape,"input")
        att_weight = tf.linalg.matmul(input,self.weight,transpose_b=True)  # Fea x Mem^T, (TxC) x (CxM) = TxM
        print(att_weight.shape)
        att_weight = tf.nn.softmax(att_weight)
  # TxM
        # ReLU based shrinkage, hard shrinkage for positive value
        if(self.shrink_thres>0):

            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            print(att_weight.shape)
            # att_weight = F.softshrink(att_weight, lambd=self.shrink_thres)
            # normalize???
            att_weight = tf.math.l2_normalize(att_weight, axis=1, epsilon=1e-12)

            print(type(att_weight))
            # att_weight = F.softmax(att_weight, dim=1)
            # att_weight = self.hard_sparse_shrink_opt(att_weight)
        mem_trans = self.weight  # Mem^T, MxC
        print(mem_trans.shape)
        output = tf.linalg.matmul(att_weight,mem_trans)  # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC
        return {'output': output, 'att': att_weight}  # output, att_weight

    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}'.format(
            self.mem_dim, self.fea_dim is not None
        )


# NxCxHxW -> (NxHxW)xC -> addressing Mem, (NxHxW)xC -> NxCxHxW
class MemModule(tf.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)

    def __call__(self, input):
        s = input.shape
        l = len(s)

       
        #x = x.contiguous()
        #x = tf.reshape(x,(-1, s[1]))
        #
        y_and = self.memory(input)
        #
        y = y_and['output']
        att = y_and['att']

        return {'output': y, 'att': att}

# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (tf.nn.relu(input-lambd) * input) / (tf.math.abs(input - lambd) + epsilon)
    return output


def video_to_frame(train_or_test):
    print(train_or_test)
    video_path = os.path.join(video_root_path,'test_segments_LV')
    print(video_path)
    frame_path = os.path.join(video_root_path, '{}_frames_new'.format(train_or_test))
    os.makedirs(frame_path, exist_ok=True)
    v=1
    f = 1
    for video_file in os.listdir(video_path):
        print('Done for video ',v)
        if video_file.lower().endswith(('.wmv', '.avi')):
            print('==> ' + os.path.join(video_path, video_file))
            #vid_frame_path = os.path.join(frame_path, os.path.basename(video_file).split('.')[0])
            vid_frame_path = frame_path
            os.makedirs(vid_frame_path, exist_ok=True)
            vidcap = skvideo.io.vreader(os.path.join(video_path, video_file))
            for image in vidcap:
              #image = resize(image, size, mode='reflect'):
              imsave(os.path.join(vid_frame_path, '{:05d}.jpg'.format(f)), image) 
              f = f+1   # save frame as JPEG file
        v=v+1

# avenue
#video_to_frame('test_segments_LV')
def cosine_sim(self, x1, x2):
    num = tf.linalg.matmul(x1, tf.transpose(x2, perm=[0, 1, 3, 2]), name='attention_num')
    denom =  tf.linalg.matmul(x1**2, tf.transpose(x2, perm=[0, 1, 3, 2])**2, name='attention_denum')
    w = (num + 1e-12) / (denom + 1e-12)
    return w


train_frames=[]

frame_list=[]

vid_root_train = "D:\\Paper\\dataset\\avenue\\training\\frames"
print(os.listdir(vid_root_train))

folder_list = os.listdir(vid_root_train)
folder_list.sort()
image_set_list=[]

for folder in folder_list:
    image_temp_path = os.path.join(vid_root_train,folder)
    #print(image_temp_path)
    image_temp=os.listdir(image_temp_path)
    image_temp.sort()
    image_set_list+=[image_temp_path+"\\"+i for i in image_temp]


#print(image_set_list)



for paths in image_set_list: #total frame 15328
  #print(paths)
  frame=cv2.imread(paths)  #path change
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  frame=cv2.resize(frame,(227,227),interpolation=cv2.INTER_AREA) #size
  frame=np.expand_dims(frame,2)
  train_frames.append(frame)
  #print(frame)
train_frames=np.array(train_frames)


X_train=[]
for i in range(0,len(train_frames),10): # total frame # frames in segment
    X_train.append(np.array(train_frames[i:i+10]))


X_train=np.array(X_train[:-1])
print(X_train.shape)



train_frames=[]
X_train=np.reshape(X_train,(287,227,227,10,1)) #change 1532
X_train.shape
np.save('D:\\Paper\\X_trainaven.npy',X_train)
X_train=(X_train-np.mean(X_train))/np.std(X_train)



img=Input(shape=(227,227,10,1))

model=Conv3D(filters=128,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',activation='tanh')(img)
model=Conv3D(filters=64,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='tanh')(model)

model=ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,padding='same',dropout=0.4,recurrent_dropout=0.3,return_sequences=True)(model)
#model_s=model
model=Conv3D(filters=64,kernel_size=(3,3,1),strides=(1,1,1),padding='same',activation='relu')(model)
model=Conv3D(filters=64,kernel_size=(3,3,1),strides=(1,1,1),padding='same',activation='relu')(model)
#model=add([model,model_s])
model=Activation('relu')(model)
#model_s=model
model=Conv3D(filters=64,kernel_size=(3,3,1),strides=(1,1,1),padding='same',activation='relu')(model)
model=Conv3D(filters=64,kernel_size=(3,3,1),strides=(1,1,1),padding='same',activation='relu')(model)
#model=add([model,model_s])
model=Activation('relu')(model)
#model_s=model
model=Conv3D(filters=64,kernel_size=(3,3,1),strides=(1,1,1),padding='same',activation='relu')(model)
model=Conv3D(filters=64,kernel_size=(3,3,1),strides=(1,1,1),padding='same',activation='relu')(model)
#model=add([model,model_s])
model=Activation('relu')(model)
model_w=ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,return_sequences=True, padding='same',dropout=0.5)
model = model_w(model)


N = 2000
shrink_thres=0.0025
#w_memory = tf.Variable(trainable=True,shape=tf.TensorShape(N,64))
#print(w_memory.shape,"w_mem_shape")
#print(model.shape)
model = MemModule(mem_dim=N, fea_dim=64, shrink_thres =shrink_thres)(model)
"""
cosim = cosine_sim(x1=model, x2=w_memory) # Eq.5
attention = tf.nn.softmax(cosim) # Eq.4


lam = 1 / N # deactivate the 1/N of N memories.

addr_num = tf.keras.activations.relu(attention - lam) * attention
addr_denum = tf.abs(atteniton - lam) + 1e-12
memory_addr = addr_num / addr_denum


renorm = tf.clip_by_value(memory_addr, 1e-12, 1-(1e-12))

z_hat = tf.linalg.matmul(renorm, w_memory)

"""
model=Conv3DTranspose(filters=128,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='tanh')(model['output'])
model=Conv3DTranspose(filters=1,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',activation='tanh')(model)

model=Model(img,model)
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
model.summary()


his=model.fit(X_train,X_train,epochs=100,batch_size=16)

h=pd.DataFrame(his.history)
plt.plot(h.index.tolist(),h['loss'])

model.save('D:\\Paper\\Deep\\ae_res_mp_avenue.h5') #Change here
"""
test_frames=[]
frame_list=[]
for i in range(1,15324): #total test frames
    print(i)    
    #frame=cv2.imread('drive//My Drive//avenue_new//testing_frames_frames_new//'+str(i)+'.jpg',0) #path
    frame=cv2.imread('D:\\HAR DATASETS av vio etc\\HAR Dataset\\CV\\Fold 1\\test_segments_frames_new'+'\\{:05d}.jpg'.format(i),0)  #path change
    #frame=cv2.imread('D:\\HAR DATASETS av vio etc\\HAR Dataset\\CV\\Fold 1\\Train001'+'\\{:05d}.tif'.format(i),0)  #path change
    frame=cv2.resize(frame,(227,227),interpolation=cv2.INTER_AREA)
    test_frames.append(frame)
test_frames=np.array(test_frames)

X_test=[]
for i in range(0,len(test_frames),10):
    X_test.append(test_frames[i:i+10])
X_test=np.array(X_test[:-1])
#X_test=np.array(X_test)
X_test.shape

X_test=np.reshape(X_test,(1532,227,227,10,1)) # change 1532
X_test.shape
np.save('D://HAR DATASETS av vio etc//HAR Dataset//CV//Fold 1//X_testaven.npy',X_test)
X_test=np.load('D://HAR DATASETS av vio etc//HAR Dataset//CV//Fold 1//X_testaven.npy')
X_test=(X_test-np.mean(X_test))/np.std(X_test)

from keras.models import load_model

model=load_model('D://HAR DATASETS av vio etc//HAR Dataset//CV//Fold 1//ae_res_mp_avenueold.h5') #Change here

X_pred=model.predict(X_test)
X_pred.shape

def mean_squared_loss(x1,x2):

	diff=x1-x2
	a,b,c,d=diff.shape
	n_samples=a*b*c*d
	sq_diff=diff**2
	Sum=sq_diff.sum()
	dist=np.sqrt(Sum)
	mean_dist=dist/n_samples

	return mean_dist

mse=[]
for i in range(len(X_test)):
  print(i)
  mse.append(mean_squared_loss(X_test[i],X_pred[i]))
  
  mse=np.array(mse)
  reg_score=1-((mse-mse.min())/mse.max())

norm_score=[]
for i in reg_score:
    norm_score.extend([i for j in range(10)])
norm_score.extend([reg_score[1531],reg_score[1531],reg_score[1531],reg_score[1531]])
norm_score=np.array(norm_score)
norm_score.shape



np.save('D://HAR DATASETS av vio etc//HAR Dataset//CV//Fold 1//without_res_regularityscore//norm_score_res_mp_avenue.npy',norm_score)
score=np.load('D://HAR DATASETS av vio etc//HAR Dataset//CV//Fold 1//without_res_regularityscore//norm_score_res_mp_avenue.npy')
gt=np.load('D:\\gan\\avenue\\ground_truth.npy')
norm_score=np.load('D://HAR DATASETS av vio etc//HAR Dataset//CV//Fold 1//without_res_regularityscore//norm_score_res_mp_avenue.npy')

c=['b' if i==0 else 'r' for i in gt]
plt.scatter([i for i in range(len(gt))],norm_score,color=['b' if i==0 else 'r' for i in gt])

from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,confusion_matrix
auc=0
thres=0
i=0.7
while i<1:
    print(i)
    y_pred=[1 if x<i else 0 for x in norm_score]
    y_pred=np.array(y_pred)
    a=roc_auc_score(gt,y_pred)
    if a>auc:
        auc=a
        thres=i
    i+=0.0001
    
print('For threshold={} AUC={}'.format(thres,auc))

y_pred=[1 if x<0.8337 else 0 for x in norm_score] #highest auc
y_pred=np.array(y_pred)
a=roc_auc_score(gt,y_pred)
cm=confusion_matrix(gt,y_pred)
acc=accuracy_score(gt,y_pred)

#import cv2
#vidcap = cv2.VideoCapture('01.avi')
#success,image = vidcap.read()
#count = 0
#while success:
#  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
#  success,image = vidcap.read()
#  print('Read a new frame: ', success)
#  count += 1
  
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train)
ntrain = scaler.transform(train)
ntest = scaler.transform(test)
"""

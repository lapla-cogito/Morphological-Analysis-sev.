#import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import sys
import h5py
import math
import flask
import random
import numpy.random as nr
from __future__ import print_function
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_uniform
from keras.initializers import uniform
from keras.optimizers import RMSprop
from keras.utils import np_utils


with open('hogehoge.csv',  encoding="shift-jis") as file:
    inp = csv.reader(file)
    data = [ i for i in inp]

#宣言
gabword=[]
mon = np.array(data)
mon2=np.r_[mon[:,0]]
words = sorted(list(set(mon2)))
cou = np.zeros(len(words))
ma=10
windex = dict((i, j) for j, i in enumerate(words))
indexw = dict((i, j) for i, j in enumerate(words))

#語数を数える
for i in range(0, len(mon2)):
    cou[windex[mon2[i]]]+=1

#文章中にあまり出てこない単語はゴミなので弾く(NGワードに書き換える)
for i in range(0, len(words)):
  if cou[i] <= 3 :
    gabword.append(words[i])
    words[i] = 'gabage'

words = sorted(list(set(words)))
windex = dict((i, j) for j, i in enumerate(words))
indexw = dict((i, j) for i, j in enumerate(words))

#訓練データの生成
monn=np.zeros((len(mon2),1),dtype=int)

for i in range(0,len(mon2)):
  if mon2[i]in windex:
    monn[i,0]=windex[mon2[i]]
  else:
    monn[i,0]=windex['gabage']
    
monlen=len(monn)-ma
train=[]
target=[]

for i in range(ma,monlen):
  train.append(monn[i])
  target.extend(monn[i-ma:i])
  target.extend(monn[i+1:i+1+ma])
  
xtrain=np.array(train).reshape(len(train),1)
ytrain=np.array(target).reshape(len(train),2*ma)
xy=zip(xtrain,ytrain)
nr.seed(12345)
nr.shuffle(xy)
xtrain,ytrain=zip(*xy)
xtrain=np.array(xtrain).reshape(len(train),1)
ytrain=np.array(ytrain).reshape(len(train),2*ma)

#ニューラルネットワークの構築
class neural:
  def init(self,inpu,outp):
    self.inpu=inpu
    self.outp=outp
  def createmodel(self):
    model = Sequential()
    model.add(Embedding(self.input_dim, self.output_dim, input_length=1, embeddings_initializer=uniform(seed=22222222)))
    model.add(Flatten())
    moddel.add(Dense(self.inpu,use_bias=False,kernel_initializer=glorot_uniform(seed=22222222)))
    mode.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy",optimizer="RMSprop",metrics=['categorical_crossentropy'])
    return model
  def train(self,xtrain,ytrain,batch_size,epochs,ma,emb_param):
    earlystop=Earlestopping(monitor='categorical_crossentropy',patience=1,verbose=1)
    model=self.vreatemodel()
    model.fit(xtrain,ytrain,batch_size=batch_size,epochs=epochs,verbose=1,shuffle=True,callbacks=[earlystop],validation_split=0.0)
    return model
  
vec_dim=100
epochs=10
batch_size=200
inpu=len(words)
outp=vec_dim
prediction=neural(inpu,outp)
row=ytrain.shape[0]
emb_param='param_skip_gfram_2_1.hdf5'
rowmon=np.zeros((row,inpu),dtype='int8')
for i in range(0,row):
  for j in range(0,2*ma):
    rowmon[i,ytrain[i,j]]=1
    
xtrain=xtrain.reshape(row,1)
model=neural.train(xtrain,ytrain,batch_size,epochs,ma,emb_param)
model.save_weights(emb_param)

param_test=model.get_weights()
param=param_test[0]
word0='下人'
word1='人間'
word2='老婆'
vec0=param[windex[word0],:]
vec1=param[windex[word1],:]
vec2=param[windex[word2],:]

vec=vec0+vec1+vec2
vecnor=math.sqrt(np.dot(vec,vec))
wordli=[windex[word0],windex[word1],windex[word2]]
dis=-1.0
fimon=0

for i in range(0,5):
  dis=-1.0
  fimon=0
  for j in range(0,len(words)):
    if j not in wordli:
      dis0=np.dot(vec,param[i,:])
      dis0=dis0/vecnor/math.sqrt(np.dot(param[i,:],param[i,:]))
      if dis<dis0:
        dis=dis0
        fimon=i
  print('第'+str(j+1)+'番目の候補=')
  print('類似度=',dis,' ',m,' ',indexw[fimon])
  wordli.append(fimon)

#import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import sys
import h5py
import math
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


#CSVファイルを読み込む
inp = csv.reader(open('hogehoge.csv','r'))

#宣言
data = [ i for i in inp]
gabword=[]
mon = np.array(data)
words = sorted(list(set(mon)))
cou = np.zeros(len(words))
ma=10
windex = dict((i, j) for j, i in enumerate(words))
indexw = dict((i, j) for i, j in enumerate(words))

#語数を数える
for i in range(0, len(words)):
    cou[windex[mon[j]]]+=1

#文章中にあまり出てこない単語はゴミなので弾く(NGワードに書き換える)
for i in range(0, len(words)):
  if cnt[i] <= 3 :
    gabword.append(words[i])
    words[i] = 'gabgabeidjdkdjekejdhogehogelatteukkuku'

words = sorted(list(set(words)))
windex = dict((i, j) for j, i in enumerate(words))
indexw = dict((i, j) for i, j in enumerate(words))

#訓練データの作成
monn=np.zeros((len(mon),1),dtype=int)
for i in range(0,len(mon)):
  if mon[i]in windex:
    monn[i,0]=windex[mon[i]]
  else:
    monn[i,0]=windex['gabgabeidjdkdjekejdhogehogelatteukkuku']
    
monlen=len(monn)-ma
target=[]


for i in range(ma,monlen):
  

#ニューラルネットワークの構築
class neural:
  def init(self,inpu,outp):
    self.inpu=inpu
    self.outp=outp

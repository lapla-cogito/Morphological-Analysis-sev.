#import
import tensorflow as tf
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
import csv
import pandas as pd
import random
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

#input data
inp = csv.reader(open('hogehoge.csv','r'))

data = [ i for i in inp]
gabword=[]
mon = np.array(data)
words = sorted(list(set(mon)))
cou = np.zeros(len(words))
windex = dict((i, j) for j, i in enumerate(words))
indexw = dict((i, j) for i, j in enumerate(words))

#count
for i in range(0, len(words)):
    cou[windex[mon[j]]]+=1
    
for i in range(0, len(words)):
  if cnt[i] <= 3 :
    gabword.append(words[i])
    words[i] = 'gabgabeidjdkdjekejdhogehogelatte'

words = sorted(list(set(words)))
windex = dict((i, j) for j, i in enumerate(words))
indexw = dict((i, j) for i, j in enumerate(words))

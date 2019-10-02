import numpy as np
import random
import pandas as pd

##
import tensorflow as tf
from tqdm import tqdm
import os
from keras.optimizers import *
from keras.layers import *
##

import scipy
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint

import train as tr

test_data = 'testdata'
def test():
    test_dat = []
    for i in tqdm(os.listdir(test_data)):
        path = os.path.join(test_data, i)
        content = pd.read_csv(path,header=None, prefix='COLUMN', skiprows=1)
        np.array(content, dtype='float32')
        content = content.values
        content = np.delete(content, [0,1,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35], 1)
        content = content/10000
        for a in content:
            test_dat.append([a])
    return test_dat
#Load model
modelpath = 'MLP.gokselmodel.best.hdf5'
testdata = test()
model = tr.createModel()
model.load_weights(modelpath)
test_images = np.array([i[0] for i in testdata])
def predict():
    for i in range(len(test_images)):
        img = np.asarray(np.reshape(test_images[i], (1,14)))
        modelout = model.predict(img)
        print(modelout)
        if np.argmax(modelout) == 1:
            print("Open")
        else:
            print("Close")
    print(len(test_images))

if __name__ == '__main__':
    predict()


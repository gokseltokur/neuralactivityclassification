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

train_data = 'traindata'
test_data = 'testdata'

def label(csv):
    label = csv.split('.')[0]
    if label == 'closed':
        ohl = np.array([1,0])
    elif label == 'opened':
        ohl = np.array([0,1])
    return ohl

def traindatalabel():
    train = []
    for i in tqdm(os.listdir(train_data)):
        path = os.path.join(train_data, i)
        content = pd.read_csv(path,header=None, prefix='COLUMN', skiprows=1)
        np.array(content, dtype='float32')
        content = content.values
        content = np.delete(content, [0,1,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35], 1)
        content = content/10000        
        for a in content:
            train.append([a, label(i)])
    #random.shuffle(train)
    return train

def testdatalabel():
    test = []
    for i in tqdm(os.listdir(test_data)):
        path = os.path.join(test_data, i)
        content = pd.read_csv(path,header=None, prefix='COLUMN', skiprows=1)
        np.array(content, dtype='float32')
        content = content.values
        content = np.delete(content, [0,1,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35], 1)
        content = content/10000       
        for a in content:
            test.append([a, label(i)])
    return test

def createModel():
    model = Sequential()
    '''
    model.add(Dense(512, activation = 'relu', input_shape = (train_data.shape[1], ))) ####### 14 yerine x.shape[1] OLABILIR DIKKAT ET
    model.add(Dense(4, activation = 'sigmoid'))
    '''
    model.add(Dense(1000, activation = 'relu', input_shape = (14, )))
    model.add(Dropout(0.2))
    model.add(Dense(1000, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1000, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation = 'sigmoid'))
    return model

def train():
    print(tf.__version__)

    trainData = traindatalabel()
    testData = testdatalabel()
    
    train_data = np.array([i[0] for i in trainData])
    train_labels = np.array([i[1] for i in trainData])
    test_data = np.array([i[0] for i in testData])
    test_labels = np.array([i[1] for i in testData])

    #(x_train, x_test) = train_data[:1100], test_data[1100:]
    #(y_train, y_test) = train_labels[:1100], test_labels[1100:]

    

    print('\n\nasd')
    print(test_data)
    print(train_data)
    print(train_labels)

    
    
    #scaler = MinMaxScaler()
    #scaler.fit()
    # BU KISIM RESHAPE YAPIYOR KUCULTURUYOR BURADA SIKINTI CIKABILIR DIKKAT ET
    """
    train_data = np.array([i[0] for i in trainData]).reshape(-1,64,64,1)
    train_labels = np.array([i[1] for i in trainData])
    test_data = np.array([i[0] for i in testData]).reshape(-1,64,64,1)
    test_labels = np.array([i[1] for i in testData])
    """
    
    class_names = ['closed', 'opened']
    num_classes = len(class_names)

    model = createModel()
     # loss eeg de binary_crossentropy ??? rectclassifier da categorical_crossentropy
    optimizer = Adam(lr=1e-3)
    model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    checkpointer = ModelCheckpoint(filepath = 'MLP.gokselmodel.best.hdf5', verbose = 1, save_best_only = True)
    #model.fit(x=train_data, y=train_labels, epochs = 10, batch_size=256, validation_split = 0.1, callbacks = [checkpointer], verbose = 2)
    model.fit(x=train_data, y=train_labels, epochs = 100, batch_size=256, validation_split = 0.1, callbacks = [checkpointer], verbose = 2)

    print('qqqqqqqqqqqqqqqqqqqqqqqq')
    loss, accuracy = model.evaluate(test_data, test_labels, verbose=1)
    #loss, accuracy = model.evaluate(test_data, test_labels)
    print('Test accuracy: ', accuracy)
    print('Test loss: ', loss)

    # 
    modelpath = 'model/gkslmodel.hdf5'
    model.save(modelpath)
    #


    print('qqqqqqqqqqqqqqqqqqqqqqqq')
    print(test_data[0])
    print(test_labels[0])
    ii = [4396.922969,5474.871661,4427.692199,5139.487054,4381.538354,4296.922972,3989.743492,4430.256302,4450.256301,4748.717833,4195.897333,4059.999901,5767.692167,4508.717838]
    asd = np.asarray(np.reshape(ii, (1,14)))
    asd = asd/10000
    print(asd)
    predict = model.predict(asd)

    print(predict)



if __name__ == '__main__':
    train()
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

def onehotlabel(csv):
    label = csv.split('.')[0]
    if label == 'closed':
        ohl = np.array([1,0])
    elif label == 'opened':
        ohl = np.array([0,1])
    return ohl

def traindatalabel():
    train_images = []
    for i in tqdm(os.listdir(train_data)):
        path = os.path.join(train_data, i)
        #img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # 
        #img = cv2.resize(img, (64, 64))
        content = pd.read_csv(path,header=None, prefix='COLUMN', skiprows=1)
        np.array(content, dtype='float32')
        content = content.values
        content = np.delete(content, [0,1,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35], 1)

        '''
        #Normalization of Data
        
        scaler = MinMaxScaler()
        scaler.fit(content)
        content_new = scaler.transform(content)
        dataMean = content.mean()
        dataStd = content.std()
        content = (content-dataMean)/dataStd
        '''
        content = content/10000
        
        for a in content:
            train_images.append([a, onehotlabel(i)])
    #random.shuffle(train_images)
    return train_images

def testdatalabel():
    print('\n\nasd')
    test_images = []
    for i in tqdm(os.listdir(test_data)):
        path = os.path.join(test_data, i)
        #img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # 
        #img = cv2.resize(img, (64, 64))
        content = pd.read_csv(path,header=None, prefix='COLUMN', skiprows=1)
        np.array(content, dtype='float32')
        content = content.values
        content = np.delete(content, [0,1,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35], 1)
        print("\n")
        print(path)
        content = content/10000
        
        for a in content:
            test_images.append([a, onehotlabel(i)])
    return test_images

def createModel():
    model = Sequential()
    '''
    model.add(Dense(512, activation = 'relu', input_shape = (train_images.shape[1], ))) ####### 14 yerine x.shape[1] OLABILIR DIKKAT ET
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

    trainimages = traindatalabel()
    testimages = testdatalabel()
    
    train_images = np.array([i[0] for i in trainimages])
    train_labels = np.array([i[1] for i in trainimages])
    test_images = np.array([i[0] for i in testimages])
    test_labels = np.array([i[1] for i in testimages])

    #(x_train, x_test) = train_images[:1100], test_images[1100:]
    #(y_train, y_test) = train_labels[:1100], test_labels[1100:]

    

    print('\n\nasd')
    print(test_images)
    print(train_images)
    print(train_labels)

    
    
    #scaler = MinMaxScaler()
    #scaler.fit()
    # BU KISIM RESHAPE YAPIYOR KUCULTURUYOR BURADA SIKINTI CIKABILIR DIKKAT ET
    """
    train_images = np.array([i[0] for i in trainimages]).reshape(-1,64,64,1)
    train_labels = np.array([i[1] for i in trainimages])
    test_images = np.array([i[0] for i in testimages]).reshape(-1,64,64,1)
    test_labels = np.array([i[1] for i in testimages])
    """
    
    class_names = ['closed', 'opened']
    num_classes = len(class_names)

    model = createModel()
     # loss eeg de binary_crossentropy ??? rectclassifier da categorical_crossentropy
    optimizer = Adam(lr=1e-3)
    model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    checkpointer = ModelCheckpoint(filepath = 'MLP.weights.best.hdf5', verbose = 1, save_best_only = True)
    #model.fit(x=train_images, y=train_labels, epochs = 10, batch_size=256, validation_split = 0.1, callbacks = [checkpointer], verbose = 2)
    model.fit(x=train_images, y=train_labels, epochs = 100, batch_size=256, validation_split = 0.1, callbacks = [checkpointer], verbose = 2)

    print('qqqqqqqqqqqqqqqqqqqqqqqq')
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=1)
    #loss, accuracy = model.evaluate(test_images, test_labels)
    print('Test accuracy: ', accuracy)
    print('Test loss: ', loss)

    # 
    modelpath = 'model/gkslmodel.hdf5'
    model.save(modelpath)
    #


    print('qqqqqqqqqqqqqqqqqqqqqqqq')
    print(test_images[0])
    print(test_labels[0])
    ii = [4396.922969,5474.871661,4427.692199,5139.487054,4381.538354,4296.922972,3989.743492,4430.256302,4450.256301,4748.717833,4195.897333,4059.999901,5767.692167,4508.717838]
    asd = np.asarray(np.reshape(ii, (1,14)))
    asd = asd/10000
    print(asd)
    predict = model.predict(asd)

    print(predict)



if __name__ == '__main__':
    train()
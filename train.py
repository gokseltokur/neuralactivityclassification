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
        ohl = np.array([1,0,0,0])
    elif label == 'opened':
        ohl = np.array([0,1,0,0])
    elif label == 'text':
        ohl = np.array([0,0,1,0])
    elif label == 'cart':
        ohl = np.array([0,0,0,1])
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


        #Normalization of Data
        '''
        scaler = MinMaxScaler()
        scaler.fit(content)
        content_new = scaler.transform(content)
        dataMean = content.mean()
        dataStd = content.std()
        content = (content-dataMean)/dataStd
        '''
        for a in content:
            train_images.append([a, onehotlabel(i)])
    random.shuffle(train_images)
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
        
        for a in content:
            test_images.append([a, onehotlabel(i)])
    return test_images

def createModel(train_images):
    model = Sequential()
    '''
    model.add(Dense(512, activation = 'relu', input_shape = (train_images.shape[1], ))) ####### 14 yerine x.shape[1] OLABILIR DIKKAT ET
    model.add(Dense(4, activation = 'sigmoid'))
    '''
    model.add(Dense(1000, activation = 'relu', input_shape = (train_images.shape[1], )))
    model.add(Dropout(0.2))
    model.add(Dense(1000, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1000, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation = 'softmax'))


    return model

def train():
    print(tf.__version__)

    trainimages = traindatalabel()
    testimages = testdatalabel()
    
    train_images = np.array([i[0] for i in trainimages])
    train_labels = np.array([i[1] for i in trainimages])
    test_images = np.array([i[0] for i in testimages])
    test_labels = np.array([i[1] for i in testimages])

    (x_train, x_test) = train_images[:1100], test_images[1100:]
    (y_train, y_test) = train_labels[:1100], test_labels[1100:]


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
    
    class_names = ['closed', 'opened', 'text', 'cart']
    num_classes = len(class_names)

    model = createModel(train_images)
     # loss eeg de binary_crossentropy ??? rectclassifier da categorical_crossentropy
    optimizer = Adam(lr=1e-3)
    model.compile(loss = 'categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    checkpointer = ModelCheckpoint(filepath = 'MLP.weights.best.hdf5', verbose = 1, save_best_only = True)
    model.fit(train_images, train_labels, epochs = 10, batch_size=256, validation_split = 0.1, callbacks = [checkpointer], verbose = 2, shuffle = True)

    print(x_test)
    print('qqqqqqqqqqqqqqqqqqqqqqqq')
    print(y_test)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print('Test accuracy: ', accuracy)
    print('Test loss: ', loss)
    print('qqqqqqqqqqqqqqqqqqqqqqqq')
    print(x_test[0])
    asd = np.asarray(np.reshape(x_test[0], (1,14)))
    print(asd)
    predict = model.predict(asd)

    print(predict)



if __name__ == '__main__':
    train()
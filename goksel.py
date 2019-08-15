import numpy as np
import random
import pandas as pd
import scipy
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
'''
csvPath = 'dataset.csv'
content = pd.read_csv(csvPath,header=None, prefix='COLUMN', skiprows=1)
np.array(content, dtype='float32')
content=content.values

content = np.delete(content, [0,1,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35], 1)
'''

fname = "EEG Eye State.txt"
with open(fname) as f:
    content = f.readlines()

content = [x.strip() for x in content] 
content = [x.split(",") for x in content]

content = np.array(content, dtype = 'float32')

print(content)

#print(content[0])

# random.shuffle(content) #Shuffling the dataset

score_p = [] # Storing results of algorithms

x = content[:, :-1]
y = np.array(content[:, -1], dtype = 'int32')

#print(x[0])

# STATISTICAL APPROACH

X_columns = ['mean', 'standard deviation', 'kurt', 'skewness']
Y_columns = ['label']

X = pd.DataFrame(columns = X_columns)
Y = pd.DataFrame(columns = Y_columns)

i = 0
for i in range(len(x)):
    X.loc[i] = np.array([np.mean(x[i]), np.std(x[i]), scipy.stats.kurtosis(x[i]), scipy.stats.skew(x[i])])
    Y.loc[i] = y[i]
    print(len(x))
    print(i)
    i+=1

print(X.head(n=20))


# Training on SVM
'''
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size = 0.2, random_state = 0)

clf = SVC()
clf.fit(X_train1, y_train1)

predicted = clf.predict(X_test1)

print("Accuracy = {}\nPrecision = {}\nRecall = {}\nF1 Score = {}".format(metrics.accuracy_score(y_test1, predicted), 
                                                                        metrics.precision_score(y_test1, predicted, pos_label='positive', average='micro'),
                                                                        metrics.recall_score(y_test1, predicted, pos_label='positive', average='micro'),
                                                                        metrics.f1_score(y_test1, predicted, pos_label='positive', average='micro')))

score_p.append([metrics.accuracy_score(y_test1, predicted), 
                metrics.precision_score(y_test1, predicted, pos_label='positive', average='micro'),
                metrics.recall_score(y_test1, predicted, pos_label='positive', average='micro'),
                metrics.f1_score(y_test1, predicted, pos_label='positive', average='micro')])
'''
# Directly use 14 values of EEG data and use it for prediction #
#Normalization of data
scaler = MinMaxScaler()
scaler.fit(x)
x_new = scaler.transform(x)
data_mean = x.mean()
data_std = x.std()
x = (x - data_mean)/data_std
(x_train, x_test) = x[:11000], x[11000:]
(y_train, y_test) = y[:11000], y[11000:]

# Training on Neural Networks 
# Creating Model
model = Sequential()
model.add(Dense(512, activation = 'relu', input_shape = (x.shape[1], )))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

checkpointer = ModelCheckpoint(filepath = 'MLP.weights.best.hdf5', verbose = 1, save_best_only = True)
hist = model.fit(x_train, y_train, epochs = 100, batch_size=256, validation_split = 0.1, callbacks = [checkpointer], verbose = 2, shuffle = True)

score = model.evaluate(x_test, y_test, verbose=1)
print("Accuracy: ", score[1])

predict2 = [1 if a>0.5 else 0 for a in model.predict(x_test)]
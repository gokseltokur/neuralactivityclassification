from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import scipy
import scipy.stats

def column(matrix, i):
    return [row[i] for row in matrix]

train_data = 'traindata'
test_data = 'testdata'

for i in tqdm(os.listdir(train_data)):
    path = os.path.join(train_data, i)
    path = os.path.join(train_data, i)
    #img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # 
    #img = cv2.resize(img, (64, 64))
    content = pd.read_csv(path,header=None, prefix='COLUMN', skiprows=1)
    np.array(content, dtype='float32')
    content = content.values
    content = np.delete(content, [0,1,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35], 1)
    
    data = {
            'AF3'   : pd.Series(column(content, 0)),
            'F7'    : pd.Series(column(content, 1)),
            'F3'    : pd.Series(column(content, 2)),
            'FC5'    : pd.Series(column(content, 3)),
            'T7'    : pd.Series(column(content, 4)),
            'P7'    : pd.Series(column(content, 5)),
            'O1'    : pd.Series(column(content, 6)),
            'O2'    : pd.Series(column(content, 7)),
            'P8'    : pd.Series(column(content, 8)),
            'T8'    : pd.Series(column(content, 9)),
            'FC6'    : pd.Series(column(content, 10)),
            'F4'    : pd.Series(column(content, 11)),
            'F8'    : pd.Series(column(content, 12)),
            'AF4'    : pd.Series(column(content, 13))
    }
    print(path)
    df = pd.DataFrame(data)
    print(df.mean())
    print('\nMinimum')
    print(df.min())
    print('\nMaximum')
    print(df.max())
    print('\nStandard Deviation')
    print(df.std())
    print('\n\n\n')
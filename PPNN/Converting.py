# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 23:48:49 2019

@author: User
"""

#import cv2 as cv2
import numpy as np
import pandas as pd

samples = np.loadtxt('generalsamples.data',np.float32)
responses = np.loadtxt('generalresponses.data',np.float32)

samples = np.array(samples,np.int32)
responses = np.array(responses,np.int32)

train = pd.DataFrame(data=samples[1:,1:],index=samples[1:,0],columns=samples[0,1:])
#Y_train = pd.DataFrame(data=responses[1:,1:],index=responses[1:,0],columns=responses[0,1:])

print(samples)
print(train)
#print(Y_train)
import pandas as pd
import numpy as np
data =samples
data.shape
type(data)
df=pd.DataFrame(data=data[0:,0:],
        index=[i for i in range(data.shape[0])],
        columns=['f'+str(i) for i in range(data.shape[1])])
print(df.head())

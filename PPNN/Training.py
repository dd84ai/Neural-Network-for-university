# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 22:17:28 2019

@author: User
"""

import cv2 as cv2
import numpy as np

def Train():
    #######   training part    ############### 
    samples = np.loadtxt('generalsamples.data',np.float32)
    responses = np.loadtxt('generalresponses.data',np.float32)
    responses = responses.reshape((responses.size,1))
    
    model = cv2.ml.KNearest_create()
    model.train(samples,cv2.ml.ROW_SAMPLE,responses)
    return model
model = Train()

############################# testing part  #########################
def sortSecond(val): 
    return val[1]  
def Extract():
    ############################# testing part  #########################
    
    im = cv2.imread('pi2.png')
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
    
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    arr = [2,]
    for cnt in contours:
        if cv2.contourArea(cnt)>20:
            [x,y,w,h] = cv2.boundingRect(cnt)
            if  h>8:
                cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
                roi = thresh[y:y+h,x:x+w]
                roismall = cv2.resize(roi,(10,10))
                roismall = roismall.reshape((1,100))
                roismall = np.float32(roismall)
                retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
                string = str(int((results[0][0])))
                arr.append([string,x])

    #arr = (list(reversed(arr)))
    leng = len(arr)
    i = 0
    while(i < leng):
        if (type(arr[i]) == type(1)):
            arr.pop(i)
            leng -= 1
        i += 1
                
    arr.sort(key = sortSecond) 
    print(arr)
    st = ""
    repeated = []
    notfound = True
    if (len(arr)>1):
        for value in arr:
            if (type(value) == type(1)):
                continue
            notfound = True
            for j in range(len(repeated)):
                if (abs(int(value[1])-repeated[j]) < 10):
                    notfound = False
                    break
            if (notfound):
                st += value[0]
                repeated.append(int(value[1]))

    return st
print(Extract())
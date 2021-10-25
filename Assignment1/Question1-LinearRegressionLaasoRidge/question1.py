#!/usr/bin/env python
# coding: utf-8

# In[179]:


import numpy as np
import matplotlib.pyplot as plt
from math import *
import pandas as pd
import random
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score 
np.set_printoptions(suppress=True)


# In[168]:


#function to load the data and randomise it and normalise it further
def loadData():
    df = pd.read_csv('abalone.data', sep=",", index_col=False)
    df.columns = ["sex", "length", "diameter","height","whole weight","shucked weight","viscera weight","shell weight","rings"]
    df = df.sample(frac=1).reset_index(drop=True)
    #mapping male=0, female=1, infant=2
    data=np.array(df)
    for i in range(len(data)):
        if(data[i][0]=='M'):
            data[i][0]=2
        elif(data[i][0]=='F'):
            data[i][0]=1
        else:
            data[i][0]=0
    data=np.array(data,dtype=float)
        
#     print(df)
#     print(data)
#     normalise(data)
    x=data[:,:8]
    y=data[:,-1]
#     print(x)
#     print(y)
    return x,y
    
    
    
    


# In[169]:


#to normalise data, gets big data into range
def normalise(data):
    for i in range(0,data.shape[1]-1):
        data[:,i] = ((data[:,i] - np.mean(data[:,i]))/np.std(data[:, i]))
#     print(data)


# In[170]:


#calculating the cost
def cost(x,y,theta):
    return sqrt(((np.matmul(x,theta)-y).T@(np.matmul(x,theta)-y))/(y.shape[0]))
    


# In[171]:


#performing gradient decent to find the minimal cost
def gradientDecent(x,y,theta,learningRate,epoch,testingX,testingY):
#     y=np.reshape(y,(-1,1))
#     print("shapesxxxxxxxxxxxxxxxxxxxxx")
#     print(y.shape)
#     print(x.shape)
#     print(theta.shape)
#     check=np.matmul(x,theta)
#     print("checkddd",check)
#     print((x.T@np.matmul(x,theta)).shape)
#     print((x.T@check).shape)
    m=x.shape[0]  #number of entries in data
    allJ=[]
    allJTest=[]
#     for i in range(theta.shape[0]):
#         theta[i,0]=2
    for i in range(epoch):
        tempCost=(x.T@(np.matmul(x,theta)-y))/m
#         print("tempcost",tempCost.shape)
        theta-=(learningRate)*tempCost
        allJ.append(cost(x,y,theta))
        allJTest.append(cost(testingX,testingY,theta))
    
    return theta, allJ, allJTest
        
    


# In[172]:


def train_test_split(x,y):
    trainX=[]
    testX=[]
    trainY=[]
    testY=[]
    #we have train:test = 8:2
    trainFreq=int((8/10)*x.shape[0])
    testFreq=x.shape[0]-trainFreq
    for i in range(trainFreq):
        trainX.append(x[i])
        trainY.append(y[i])
    for i in range(trainFreq,x.shape[0]):
        testX.append(x[i])
        testY.append(y[i])
    
    return np.array(trainX),np.array(trainY),np.array(testX),np.array(testY)
    


# In[173]:


#performing linear regression by invoking all the functions
x,y = loadData()
# print(x.shape)
# print(y.shape)
y=np.reshape(y,(-1,1))
print(x.shape)
print(y.shape)
x = np.hstack((np.ones((x.shape[0],1)),x)) #adding a column of 1s for matrix multiplication
trainingX,trainingY,testingX,testingY = train_test_split(x,y)

theta=np.zeros((trainingX.shape[1],1))

theta, allJ, allJTest = gradientDecent(trainingX,trainingY,theta,0.1,100000,testingX,testingY) #we have a learning rate of 0.1 and 100000 epoch

J=cost(trainingX,trainingY,theta)
JTest=cost(testingX,testingY,theta)
print("cost training: ",J)
print("parameters: ", theta)
print("cost testing",JTest)







# In[174]:


allJMean=np.array(allJ).mean(axis=0)
allJTestMean=np.mean(np.array(allJTest),axis=0)
# print(np.array(allJTest).shape)
# print(allJTestMean)
allJ=np.array(allJ)
allJTest=np.array(allJTest)
# print(allJ[:25])
plt.plot(allJ)
plt.title("RMSE vs epoch for training set")
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.show()
plt.plot(allJTest)
plt.title("RMSE vs epoch for testing set")
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.show()


# In[182]:


def ridgeRegression(x,y,learningRate,theta,L,epoch,testingX,testingY):
    
    m=x.shape[0]  #number of entries in data
    allJ=[]
    allJTest=[]
    
    for i in range(epoch):
        tempCost=(x.T@(np.matmul(x,theta)-y))/m
        tempCost+=L*np.sum(theta)
        theta-=(learningRate)*tempCost
        allJ.append(cost(x,y,theta))
        allJTest.append(cost(testingX,testingY,theta))
    
    return theta, allJ, allJTest


# In[ ]:


alphas=[0.1,0.2,0.3,0.4,0.5,0.55,0.6,0.7,0.8,0.45,0.35]
hashyAlphasInfo={}

for alpha in alphas:
    
    
    






# In[183]:


alphas = np.logspace(-4,1,1000)
model = Ridge()
grid = GridSearchCV(estimator=model,param_grid=dict(alpha=alphas),cv=5)
grid.fit(trainingX,trainingY)
L = grid.best_estimator_.alpha
print("Optimal hyperparameter: ",L)

theta=np.zeros((trainingX.shape[1],1))

theta, allJRidge, allJTestRidge = ridgeRegression(trainingX,trainingY,0.1,theta,L,100000,testingX,testingY) #we have a learning rate of 0.1 and 100000 epoch

J=cost(trainingX,trainingY,theta)
JTest=cost(testingX,testingY,theta)
print("cost training for ridge regression: ",J)
print("parameters for ridge regression: ", theta)
print("cost testing for ridge regression",JTest)

allJRidge=np.array(allJRidge)
allJTestRidge=np.array(allJTestRidge)
plt.plot(allJRidge)
plt.title("RMSE vs epoch for training set for Ridge")
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.show()
plt.plot(allJTestRidge)
plt.title("RMSE vs epoch for testing set for Ridge")
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.show()


# In[184]:


def lassoRegression(x,y,learningRate,theta,L,epoch,testingX,testingY):
    
    m=x.shape[0]  #number of entries in data
    allJ=[]
    allJTest=[]
    
    for i in range(epoch):
        tempCost=(x.T@(np.matmul(x,theta)-y))/m
        tempCost+=2*L*np.sum(theta/np.abs(theta))
        theta-=(learningRate)*tempCost
        allJ.append(cost(x,y,theta))
        allJTest.append(cost(testingX,testingY,theta))
    
    return theta, allJ, allJTest


# In[185]:


alphas = np.logspace(-4,1,1000)
model = Lasso()
grid = GridSearchCV(estimator=model,param_grid=dict(alpha=alphas),cv=5)
grid.fit(trainingX,trainingY)
L = grid.best_estimator_.alpha
print("Optimal hyperparameter: ",L)

theta=np.zeros((trainingX.shape[1],1))

theta, allJLasso, allJTestLasso = ridgeRegression(trainingX,trainingY,0.1,theta,L,100000,testingX,testingY) #we have a learning rate of 0.1 and 100000 epoch

J=cost(trainingX,trainingY,theta)
JTest=cost(testingX,testingY,theta)
print("cost training for lasso regression: ",J)
print("parameters for lasso regression: ", theta)
print("cost testing for lasso regression",JTest)

allJRidge=np.array(allJLasso)
allJTestRidge=np.array(allJTestLasso)
plt.plot(allJLasso)
plt.title("RMSE vs epoch for training set for Lasso")
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.show()
plt.plot(allJTestLasso)
plt.title("RMSE vs epoch for testing set for Lasso")
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.show()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import matplotlib.pyplot as plt
from math import *
import pandas as pd
import random
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils import read_sample
from sklearn.decomposition import PCA
from pyclustering.samples.definitions import FCPS_SAMPLES
np.set_printoptions(suppress=True)
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#function to load the data of population dataset
def loadData():
    df = pd.read_csv('population.csv', sep=",", index_col=False)
    df = df.sample(frac=1).reset_index(drop=True)
    
    toRemove=[]
    for i in df.columns:
        df.loc[df[i] == ' ?', i] = np.nan
        if(df[i].isna().sum()/len(df)>0.4):
            toRemove.append(i)
    print("Columns removed",toRemove) 
    for i in toRemove:
        df=df.drop(i,1)
        
    return df


# In[66]:


populationData=loadData()


# In[4]:


#plotting histogram
for i in populationData.columns:
    freq=populationData[i].value_counts()
    values=[]
    frequency=[]
    for j in freq.index:
        values.append(j)
        frequency.append(freq[j])
#     width = np.diff(values).min()
    plt.bar(values,frequency,align='center')
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title(i)
    plt.show()
#     print("Above graph's x axis values are ",values)


# In[67]:


#droping features having more than 80% of the frequency in one feature only
toRemove=[]
for i in populationData.columns:
    freq=populationData[i].value_counts()
    maxNum=0
    total=0
    for j in freq.index:
        maxNum=max(freq[j],maxNum)
        total+=freq[j]
    if(maxNum/total>0.8):
        toRemove.append(i)
        populationData=populationData.drop(i,1)

print("Columns removed are ",toRemove)
print(populationData.shape)
    


# In[68]:


modeOfColumns={} #calculating mode of columns and storing for later use

for i in populationData.columns:
    
    mode=populationData[i].mode()
    modeOfColumns[i]=mode[0]
    populationData[i].fillna(mode[0],inplace=True)


# In[49]:


print(modeOfColumns)


# In[69]:


#bucketize
numericalColumns=['AAGE','WKSWORK']

for i in populationData.columns:
    if(i in numericalColumns):
        labels = ['lowest','low','neutral','high','highest']    #1 being the lowest and 5 being the  highest
        populationData[i+'Binned']=pd.cut(populationData[i],bins=5,precision=0,labels=labels)
#         print(populationData[i+"Binned"].value_counts())
        populationData=populationData.drop(i,1)


# In[70]:


#one hot encode
# y = pd.get_dummies(df.Countries, prefix='Country')
for i in populationData.columns:
    if(len(populationData[i].value_counts().index)==2):
        oneHot=pd.get_dummies(populationData[i], prefix=i)
        
        populationData = pd.concat([populationData, oneHot[oneHot.columns[0]]], axis=1)
        populationData = populationData.drop(i,1)
        continue
    oneHot=pd.get_dummies(populationData[i], prefix=i)
    populationData = pd.concat([populationData, oneHot], axis=1)
    populationData = populationData.drop(i,1)
print(populationData.head(10))


# In[23]:


#PCA
variance=[]
for i in range(25,46):
    pca = PCA(n_components = i, random_state = 0)
    pca.fit(populationData)
    variance.append(np.sum(pca.explained_variance_ratio_))
    print(variance[-1],i)
#     pcaData = pd.DataFrame(pca.fit_transform(populationData))
plt.plot(range(25,46),variance)
plt.xlabel("n_components")
plt.ylabel("variance")
plt.show()


# In[86]:


#choosing n_components=38 as we first got 85+ variance there
pcaPopulation = PCA(n_components = 38, random_state = 0)
pcaData = pd.DataFrame(pcaPopulation.fit_transform(populationData))


# In[25]:


print(pcaData.shape)


# In[26]:


#clustering 

loss=[]
for i in range(10,25):
    print("currently at k =",i)
    randomNums = np.random.choice(len(pcaData),i,replace=False)
    medianInit=pcaData.loc[randomNums]
    kmedians_instance=kmedians(pcaData, medianInit)
    kmedians_instance.process()
    clusters=kmedians_instance.get_clusters()
    medians=kmedians_instance.get_medians()
    lossAtK=(kmedians_instance.get_total_wce()/i)
    loss.append(lossAtK)


# In[27]:


plt.plot(range(10,25),loss,'g')
plt.xlabel("K")
plt.ylabel("Loss")
plt.show()


# In[72]:


#by elbow in graph we get best k as 12
bestK=12
randomNums = np.random.choice(len(pcaData),bestK,replace=False)
medianInit=pcaData.loc[randomNums]
kmedians_instance=kmedians(pcaData, medianInit)
kmedians_instance.process()
clustersPopulation=kmedians_instance.get_clusters()
mediansPopulation=kmedians_instance.get_medians()
lossAtK=(kmedians_instance.get_total_wce()/bestK)
print("Loss we get for chosen best k = ",lossAtK)


# In[ ]:


#Part 5


# In[29]:


#function to load the data of more_than_50k dataset 
def loadDataMore():
    df = pd.read_csv('more_than_50k.csv', sep=",", index_col=False)
    df = df.sample(frac=1).reset_index(drop=True)
    
    toRemove=[]
    for i in df.columns:
        df.loc[df[i] == ' ?', i] = np.nan
        if(df[i].isna().sum()/len(df)>0.4):
            toRemove.append(i)
    print("Columns removed",toRemove) 
    for i in toRemove:
        df=df.drop(i,1)
        
    return df


# In[43]:


moreData=loadDataMore()


# In[44]:


#plotting histogram
for i in moreData.columns:
    freq=moreData[i].value_counts()
    values=[]
    frequency=[]
    for j in freq.index:
        values.append(j)
        frequency.append(freq[j])
    plt.bar(values,frequency,align='center')
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title(i)
    plt.show()


# In[45]:


#droping features having more than 80% of the frequency in one feature only
toRemove=[]
for i in moreData.columns:
    freq=moreData[i].value_counts()
    maxNum=0
    total=0
    for j in freq.index:
        maxNum=max(freq[j],maxNum)
        total+=freq[j]
    if(maxNum/total>0.8):
        toRemove.append(i)
        moreData=moreData.drop(i,1)

print("Columns removed are ",toRemove)
print(moreData.shape)


# In[50]:


#replacing with mode values calculated earlier
for i in moreData.columns:
    moreData[i].fillna(modeOfColumns[i],inplace=True)


# In[56]:


#bucketize
numericalColumns=['AAGE','WKSWORK','DIVVAL']

for i in moreData.columns:
    if(i in numericalColumns):
        labels = ['lowest','low','neutral','high','highest']    #1 being the lowest and 5 being the  highest
        moreData[i+'Binned']=pd.cut(moreData[i],bins=5,precision=0,labels=labels)
        moreData=moreData.drop(i,1)


# In[57]:


#one hot encode
for i in moreData.columns:
    if(len(moreData[i].value_counts().index)==2):
        oneHot=pd.get_dummies(moreData[i], prefix=i)
        
        moreData = pd.concat([moreData, oneHot[oneHot.columns[0]]], axis=1)
        moreData = moreData.drop(i,1)
        continue
    oneHot=pd.get_dummies(moreData[i], prefix=i)
    moreData = pd.concat([moreData, oneHot], axis=1)
    moreData = moreData.drop(i,1)
print(moreData.head(10))


# In[58]:


#PCA
variance=[]
for i in range(25,55):
    pca = PCA(n_components = i, random_state = 0)
    pca.fit(moreData)
    variance.append(np.sum(pca.explained_variance_ratio_))
    print(variance[-1],i)
#     pcaData = pd.DataFrame(pca.fit_transform(populationData))
plt.plot(range(25,55),variance)
plt.xlabel("n_components")
plt.ylabel("variance")
plt.show()


# In[87]:


#choosing n_components=49 as we first got 85+ variance there
pcaMore = PCA(n_components = 49, random_state = 0)
pcaDataMore = pd.DataFrame(pcaMore.fit_transform(moreData))


# In[60]:


#clustering 

loss=[]
for i in range(10,25):
    print("currently at k =",i)
    randomNums = np.random.choice(len(pcaData),i,replace=False)
    medianInit=pcaData.loc[randomNums]
    kmedians_instance=kmedians(pcaData, medianInit)
    kmedians_instance.process()
    clusters=kmedians_instance.get_clusters()
    medians=kmedians_instance.get_medians()
    lossAtK=(kmedians_instance.get_total_wce()/i)
    loss.append(lossAtK)


# In[61]:


plt.plot(range(10,25),loss,'g')
plt.xlabel("K")
plt.ylabel("Loss")
plt.show()


# In[64]:


#by elbow in graph we get best k as 14
bestK=14
randomNums = np.random.choice(len(pcaData),bestK,replace=False)
medianInit=pcaData.loc[randomNums]
kmedians_instance=kmedians(pcaData, medianInit)
kmedians_instance.process()
clustersMore=kmedians_instance.get_clusters()
mediansMore=kmedians_instance.get_medians()
lossAtK=(kmedians_instance.get_total_wce()/bestK)
print("Loss we get for chosen best k = ",lossAtK)


# In[ ]:


#part 6


# In[80]:


hashy = {}
for i in range(len(clustersPopulation)):
    for j in clustersPopulation[i]:
        hashy[j] = i

array = [0]*(len(hashy))
for i in hashy:
    array[i] = hashy[i]

plt.scatter(pcaData.to_numpy()[:,0],pcaData.to_numpy()[:,1],c = array, edgecolor = 'none', alpha = 0.5, cmap = plt.cm.get_cmap("Spectral",12))
plt.colorbar()


# In[78]:


hashy = {}
for i in range(len(clustersMore)):
    for j in clustersMore[i]:
        hashy[j] = i

array = [0]*(len(hashy))
for i in hashy:
    array[i] = hashy[i]

plt.scatter(pcaDataMore.to_numpy()[:,0],pcaDataMore.to_numpy()[:,1],c = array, edgecolor = 'none', alpha = 0.5, cmap = plt.cm.get_cmap("Spectral",14))
plt.colorbar()


# In[ ]:


#6.3


# In[89]:


features = pd.DataFrame(pcaPopulation.components_,columns = populationData.columns).T
print(features[0].sort_values())


# In[ ]:


#6.4


# In[90]:


features = pd.DataFrame(pcaMore.components_,columns = moreData.columns).T
print(features[0].sort_values())


# In[ ]:





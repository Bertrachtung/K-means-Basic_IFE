#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:47:42 2019
Using ORB in opencv, good at recognize rigid body like fighter-jet
@author: tsao
"""

import cv2
import numpy as np
import os
import shutil
from skimage import io
from sklearn.decomposition import PCA
 
def distEclud(vecA, vecB):
    from scipy.spatial.distance import pdist
    X=np.vstack([vecA,vecB])
    return pdist(X)

def randCent(dataSet, k):
    n = np.shape(dataSet)[1]#矩阵列数
    m = np.shape(dataSet)[0]
    centroids = np.mat(np.zeros((k,n)))
    #for j in range(n):
        #minJ = min(dataSet[:,j])
        #rangeJ = float(max(np.array(dataSet)[:,j]) - minJ)
        #centroids[:,j] = minJ + rangeJ * np.random.rand(k,1)
    randPts = np.random.randint(m,size = k)
    for j in range(k):
        centroids[j,0] = randPts[j]
        centroids[j,:] = dataSet[randPts[j],:]
    return centroids

def extract_features(image_path, vector_size = 32):
    image = io.imread(image_path, mode = 'RGB')
    image = cv2.resize(image, (512,256), interpolation=cv2.INTER_AREA)#进行了尺寸固定操作
    try:
        alg = cv2.ORB_create()
        #可选有AKAZE、KAZA、FastFeatureDetector、Brisk、xfeatures2d.SIFT、xfeatures2d.SURF
        kps = alg.detect(image)
        kps = sorted(kps, key = lambda x: -x.response)[:vector_size]
        kps, dsc = alg.compute(image, kps)
        dsc = dsc.flatten()
        needed_size=(vector_size*64)
        if dsc.size < needed_size:
            dsc = np.concatenate([dsc,np.zeros(needed_size-dsc.size)])
    except cv2.error as e:
        print ('Error',e)
        return None
    return dsc

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m =np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,2)))#create mat to assign data points ,m行每行代表一张图
                                      #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                #print(distJI)
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: 
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        #print centroids
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            centroids[cent,:] = np.mean(ptsInClust, axis=0) #assign centroid to mean 
    sumSE = sum(clusterAssment[:,1])
    return centroids, clusterAssment, sumSE
    
def show(dataSet, k, centroids, clusterAssment):
    from matplotlib import pyplot as plt  
    numSamples, dim = dataSet.shape  
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  
    for i in range(numSamples):  
        markIndex = int(clusterAssment[i, 0])  
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])  
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']  
    for i in range(k):  
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)  
    plt.show()

def load_dataset():
    path = '/Users/tsao/desktop/datasetToge/toge/'
    imlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    features = np.zeros([len(imlist), 32*64])
    for i, f in enumerate(imlist):
        features[i]=extract_features(f)
    return features


def main():
    k = 2
    data_mat = load_dataset()
    pca = PCA(n_components=40, svd_solver='full')
    data_mat = pca.fit_transform(data_mat)
    print(pca.explained_variance_ratio_)
    #my_centroids, clust_assing, my_sumSE = kMeans(data_mat,3)
    min_SumSE = np.inf
    n = np.shape(data_mat)[1]
    m = np.shape(data_mat)[0]
    min_centroids = np.zeros([k,n])
    min_clust_assing = np.zeros([m,2])
    for i in range(10):
        my_centroids, clust_assing, my_SumSE = kMeans(data_mat,k)
        if my_SumSE<min_SumSE:
            min_SumSE = my_SumSE; min_centroids = my_centroids; min_clust_assing = clust_assing
            print(my_SumSE)
    show(data_mat, k, min_centroids, min_clust_assing) 
    output(min_centroids, min_clust_assing)
    
def output(centroids ,clusterAssment):
    path = '/Users/tsao/desktop/datasetToge/'
    n = np.shape(centroids)[0]
    mpath=[0]*n
    for i in range(n):
        mpath[i]=path + str(i)
        #print(mpath[i])
        if not os.path.exists(mpath[i]):
            os.makedirs(mpath[i])
        else:
            shutil.rmtree(mpath[i])
            os.makedirs(mpath[i])
    path = '/Users/tsao/desktop/datasetToge/toge'
    m = np.shape(clusterAssment)[0]#行数
    #print(m)
    imlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    for j in range(m):
        k = clusterAssment[j,0]
        k = int(k)
        #print(j)
        #print(k)
        shutil.copy(imlist[j], mpath[k])    
    
if __name__ == '__main__':
    main()
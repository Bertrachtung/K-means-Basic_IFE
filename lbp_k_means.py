#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 00:01:41 2019
using LBP to describe the features of images，seem to be good at judging duck and fighter-jet...
@author: tsao
"""

from skimage.feature import hog
from skimage.feature import local_binary_pattern
import cv2
import numpy as np
import os
import shutil
from sklearn.decomposition import PCA
 
 
#img = cv2.cvtColor(cv2.imread('/Users/tsao/desktop/datasetToge/nature/060_0008.jpg'), cv2.COLOR_BGR2GRAY)
#print(img.shape)
#hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8), block_norm='L2-Hys')
#print(np.shape(hog_image))
#hog = cv2.HOGDescriptor()
#img_gray = img = cv2.imread('/Users/tsao/desktop/datasetToge/nature/060_0008.jpg', cv2.IMREAD_GRAYSCALE)
#feature = hog.compute(img_gray)
#print(feature)
#print(np.shape(feature))
#feature = feature.flatten()
#print(feature)
#pca = PCA(n_components=20, svd_solver='full')
#feature = pca.fit_transform(feature)
#print(pca.explained_variance_ratio_)
#hog_image = pca.fit_transform(hog_image)
#print(np.shape(hog_image))
#print(hog_image)
#print(pca.explained_variance_ratio_)

def distEclud(vecA, vecB):
    from scipy.spatial.distance import pdist
    X=np.vstack([vecA,vecB])
    return pdist(X)

def randCent(dataSet, k):
    n = np.shape(dataSet)[1]#矩阵列数
    
    m = np.shape(dataSet)[0]
    #print(m)
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

def sci_hog(image_path,wsize = (256,512)):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256,512), interpolation=cv2.INTER_AREA)
    feature = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8), block_norm='L2-Hys')
    feature = feature.flatten()
    need_size = 512
    if feature.size < need_size:
            feature = np.concatenate([feature,np.zeros(need_size-feature.size)])
    return feature
    
def sci_lbp(image_path):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256,512), interpolation=cv2.INTER_AREA)
    feature = local_binary_pattern(img, 8, 2)
    feature = feature.flatten()
    return feature

def load_dataset(feature_method):
    path = '/Users/tsao/desktop/datasetToge/bird/'
    imlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    features = np.zeros([len(imlist), 131072])
    for i, f in enumerate(imlist):
        features[i]=feature_method(f)
    return features

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
    path = '/Users/tsao/desktop/datasetToge/bird'
    m = np.shape(clusterAssment)[0]#行数
    imlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    for j in range(m):
        k = clusterAssment[j,0]
        k = int(k)
        shutil.copy(imlist[j], mpath[k])   

def main():
    k = 2
    data_mat = load_dataset(sci_lbp)
    pca = PCA(n_components=40, svd_solver='full')
    data_mat = pca.fit_transform(data_mat)
    print(pca.explained_variance_ratio_)
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

if __name__ == '__main__':
    main()
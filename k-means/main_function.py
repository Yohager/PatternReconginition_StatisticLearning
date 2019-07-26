# -*- coding: utf-8 -*-
#author: yohager
#k-means algorithm


import numpy as np
import pandas as pd
import math
import os
from sklearn import preprocessing
import random
from scipy.spatial.distance import cdist

def load_data(filepath):
    data_set = []
    try:
        with open(filepath,"r") as file:
            for line in file:
                data_set.append(line.split(","))
        #print(set([x[-1] for x in data_set]))
        mapping = {
            "Iris-virginica":0,
            "Iris-setosa":1,
            "Iris-versicolor":2
        }
        for x in data_set:
            x[-1] = "".join(x[-1].split())
            #print(x[-1])
            x[-1] = mapping.get(x[-1])
        for x in data_set:
            x[:-1] = list(map(float,x[:-1]))
        return data_set
    except OSError as reason:
        print("ERROR"+str(reason))
    
def pre_handle_data(data):
    data = np.array(data)
    data_handled = preprocessing.minmax_scale(data)
    #print(data_handled.shape)
    return data_handled


def clustering_function(data,k,max_iteration):
    data_shuffle = np.random.permutation(data)
    data_used = data_shuffle[:,:-1]
    numbers = data_used.shape[0]
    #随机选取shuffle之后的前k个向量作为初始的均值向量
    mu_vector = data_used[:k]
    #print(mu_vector)
    data_clusters = np.empty((numbers,1))
    i = 0
    while i <= max_iteration:
        i += 1
        dis_matrix = cdist(data_used,mu_vector,metric='euclidean')
        #print(dis_matrix.shape)
        for number in range(numbers):
            data_clusters[number][0] = np.argmin(dis_matrix[number])
        #data_clusters.reshape((len(data_clusters),1))
        '''
        随机均值向量后计算每一个样本到均值的距离，更新均值向量
        '''
        data_all = np.hstack([data_used,data_clusters])
        for j in range(k):
            mu_vector[j] = np.mean(data_all[np.where(data_all[:,-1] == j)])
        print(i)
    return data_clusters
    
    
    #print(data_clusters.shape)
    #print(data_used.shape)
    #print(data_all.shape)
    



if __name__ == "__main__":
    data = load_data("iris.txt")
    #print(data)
    data_handled = pre_handle_data(data)
    print(clustering_function(data_handled,4,10))

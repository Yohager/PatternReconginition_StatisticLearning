# -*- coding: utf-8 -*-
#author: yohager
#linear discriminate analysis

import numpy as np 
import pandas as pd
import math
import random
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
from matplotlib import pyplot as plt


def load_data():
    '''
    使用sklearn加载一个经典的二分类数据集数据集共计569条样本，30个属性变量，分类结果是0/1
    '''
    double_class = load_breast_cancer()
    data_x = double_class.data
    data_y = double_class.target
    return data_x,data_y

def pre_handle_data(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(data)


#二分类的LDA线性判别分析
def LDA_function(data_x,data_y):
    #number_true = str(data_y).count('1')
    #number_false = str(data_y).count('0')
    #print(number_true,number_false)
    #data_x = pre_handle_data(data_x)
    data_true = []
    data_false = []
    #后期考虑减少这里的时间复杂度
    for i in range(len(data_x)):
        if data_y[i] == 1:
            data_true.append(data_x[i])
        else:
            data_false.append(data_x[i])
    data_true = np.array(data_true)
    data_false = np.array(data_false)
    # min_number = min(data_true.shape[0],data_false.shape[0])
    # data_true = data_true[:min_number,:]
    # data_false = data_false[:min_number,:]
    print(data_true.shape,data_false.shape)
    #计算均值
    mu_1 = np.mean(data_true,axis=0)
    mu_2 = np.mean(data_false,axis=0)
    print("均值向量的大小为：",mu_1.shape,mu_2.shape)
    #计算两类各自的协方差
    s_1 = np.cov(data_true.T)
    s_2 = np.cov(data_false.T)
    print("协方差矩阵的大小为：",s_1.shape,s_2.shape)
    #类内散度矩阵的和
    s_w = s_1 + s_2
    s_b = np.dot((mu_1 - mu_2).reshape(len(mu_1),1),(mu_1 - mu_2).reshape(len(mu_1),1).T)
    #print(s_b.shape)
    #print(s_w.shape,s_b.shape)
    s_w_reverse = np.matrix(s_w).I 
    print("s_w的逆矩阵为：",s_w_reverse)
    print("均值之差为：",mu_1-mu_2)
    eig_values,eig_vectors = np.linalg.eig(np.dot(s_w_reverse,s_b))
    print(eig_values)
    result_w = eig_vectors
    #result_w_1 = s_w_reverse.dot((mu_1 - mu_2))
    print("方法一计算得到的投影方向：",result_w)
    #print("方法二计算得到的投影方向：",result_w_1)
    vector_1 = eig_vectors.T[0]
    vector_2 = eig_vectors.T[1]
    print(np.array(vector_1).reshape(2))
    #plot_pic(data_true,data_false,vector_1)


def plot_pic(data_1,data_2,vector):
    #绘制基本的数据分布
    x_1 = data_1[:,0]
    y_1 = data_1[:,1]
    x_2 = data_2[:,0]
    y_2 = data_2[:,1]
    x_3 = np.linspace(-10,10)
    y_3 = vector[1]/vector[0] * x_3
    fig = plt.figure()
    plt.title("Initial Data Scatter")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.scatter(x_1,y_1,marker='x',color='b',label='Class One')
    plt.scatter(x_2,y_2,marker='x',color='r',label='Class Two')
    plt.plot(x_3,y_3)
    plt.legend()
    plt.show()
    







if __name__ == "__main__":
    #data_x,data_y = load_data()
    #print(data_x)
    #print(data_y.shape)
    data_x = np.array([[4,2],[2,4],[2,3],[3,6],[4,4],[9,10],[6,8],[9,5],[8,7],[10,8]])
    data_y = np.array([0,0,0,0,0,1,1,1,1,1])
    LDA_function(data_x,data_y)
    
        
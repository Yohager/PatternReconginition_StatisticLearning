# -*- encoding = utf-8 -*-
#author: Yuhang Guo
'''
Using the cs229 courses in Standford
Destination:Learning the machine learning algorithms
Try to rewrite the .m matlab codes to .py python codes
'''
# 1. Linear regression
import re
import time
import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston
from sklearn import linear_model
from sklearn.model_selection import train_test_split
class Linear_regression:
    #读取.data文件的数据,保存为一个二维list的形式
    def reading_data(self,filepath,data_set):
        with open(filepath,'r') as data:
            #print(data)
            for line in data:
                # using regular expression
                #print(len(line))
                line = re.split(r'\s+',line)
                #print(len(line),line)
                if line[0] == '':
                    line.pop(0)
                #print(len(line),line)
                line.pop()
                #str2int
                line = list(map(eval,line))
                data_set.append(line)
        return data_set

    #解释变量与响应变量，训练集和测试集的划分,共计506个样本，随机划分400个作为训练集
    #波士顿房价的数据集实际上集成于sklearn，可以简单的使用load_boston()进行调用
    def feature_scalling(self,X):
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        return (X - mean) / std
    def data_divide(self,dataset):
        dataset = np.array(dataset)
        x = dataset[:,:-1]
        y = dataset[:,-1]
        x = Linear_regression.feature_scalling(self,x)
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 0)
        data_one_1 = np.ones(len(x_train))
        data_one_2 = np.ones(len(x_test))
        x_train = np.c_[data_one_1,x_train]
        x_test = np.c_[data_one_2,x_test]
        print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
        return x_train,x_test,y_train,y_test
    #梯度下降法(Gradient Descent)
    #data_x：测试集中解释变量数据；data_y：测试集中响应变量数据；
    #theta：解释变量的系数向量；alpha：学习率（步长）；MaxIteration：最大迭代次数
    def Gradient_Descent(self,data_x,data_y,theta,alpha,MaxIteration,epsilon):
        data_x = np.array(data_x)
        data_y = np.array(data_y)
        m,n = np.shape(data_x)
        print(m,n)
        counter = 0
        theta = np.array(theta)
        #begin iteration
        error_0 = 0
        diff = []
        while counter <= MaxIteration:
            counter += 1
            error_1 = 0
            diff = np.zeros(n)
            for element in diff:
                element = 0
            for i in range(m):
                for j in range(n):
                    diff[j] += (np.dot(theta,data_x[i]) - data_y[i]) * data_x[i,j]
            for a in range(n):
                theta[a] -= alpha * diff[a]
            for k in range(m):
                error_1 += (data_y[k] - np.dot(theta,data_x[k]))**2 / (2*m)
            print("error:",error_1)
            print(error_1 - error_0)
            if abs(error_1 - error_0) < epsilon:
                break
            else:
                error_0 = error_1
        return theta

    #利用梯度下降法得到的结果进行测试集数据的预测
    def Prediction(self,testing_x,theta):
        predcting_y = []
        for i in range(len(testing_x)):
            predcting_y.append(np.dot(testing_x[i],theta.T))
        print(predcting_y)
        return predcting_y
    
    #plot_1
    def plot_data(self,predicting_data,real_data):
        length = len(predicting_data)
        x_data = range(1,length+1)
        plt.figure(num = "Predicting_Data",figsize = (8,5),dpi = 120)
        x_value = list(range(0,len(predicting_data)))
        plt.grid(True)
        plt.title('True data and predict data scatter plot')
        plt.scatter(x_data,real_data,marker='*',label='True data')
        plt.scatter(x_data,predicting_data,marker='.',label='Predict data')
        plt.legend()
        plt.xlabel('data_index')
        plt.ylabel('data_value')
        plt.savefig('scatter_value.png')
        plt.show()

    #plot2
    def f(self,a,b):
        return (a,b) if a>b else(b,a)
    def plot_contract(self,predict_data,true_data):
        a = int(np.max(predict_data))
        b = int(np.max(true_data))
        c = Linear_regression.f(self,a,b)[0]
        x = range(c)
        y = range(c)
        plt.figure(num="Predicting_Data", figsize=(6, 4), dpi=120)
        plt.title('True data and predict data')
        plt.plot(x,y)
        plt.scatter(true_data,predict_data,marker='*',c='r',label='data')
        plt.savefig('zhexian.png')
        plt.show()

    def load_data(self,shuffled=True):
        data = load_boston()
        # print(data.DESCR)# 数据集描述
        X = data.data
        y = data.target
        X = Linear_regression.feature_scalling(self,X)
        y = np.reshape(y, (len(y), 1))
        if shuffled:
            shuffle_index = np.random.permutation(y.shape[0])
            X = X[shuffle_index]
            y = y[shuffle_index]  # 打乱数据
        return X, y


if __name__ == '__main__':
    test = Linear_regression
    data_set = []
    data_test = test.reading_data(test,'housing.data',data_set)
    training_x,testing_x,training_y,testing_y = test.data_divide(test,data_set)
    length = len(training_x[0])
    theta_init = np.random.random(size = length)
    alpha = 0.0005
    MaxIteration_1 = 1000
    epsilon_1 = 0.005
    #print(theta_init)
    theta_result = test.Gradient_Descent(test,training_x,training_y,theta_init,alpha,MaxIteration_1,epsilon_1)
    print(theta_result)
    result_y = test.Prediction(test,testing_x,theta_result)
    print(testing_y)
    print(result_y)
    test.plot_data(test,result_y,testing_y)
    test.plot_contract(test,result_y,testing_y)




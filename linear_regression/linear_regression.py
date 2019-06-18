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
#from matplotlib import pyplot as plt
class Linear_regression:
    #读取.data文件的数据,保存为一个二维list的形式
    def reading_data(self,filepath,data_set):
        with open(filepath,'r') as data:
            #print(data)
            for line in data:
                # 使用正则表达式清洗数据
                #print(len(line))
                line = re.split(r'\s+',line)
                #print(len(line),line)
                if line[0] == '':
                    line.pop(0)
                #print(len(line),line)
                line.insert(0,'1')
                line.pop()
                #str转int
                line = list(map(eval,line))
                data_set.append(line)
        return data_set

    #解释变量与响应变量，训练集和测试集的划分,共计506个样本，随机划分400个作为训练集
    #波士顿房价的数据集实际上集成于sklearn，可以简单的使用load_boston()进行调用
    def data_division(self,dataset):
        #print(type(dataset))
        random.shuffle(dataset)
        data_np = np.array(dataset)
        Training_x = data_np[:401,:-1]
        Training_y = data_np[:401,-1]
        Testing_x = data_np[401:,:-1]
        Testing_y = data_np[401:,-1]
        return Training_x,Training_y,Testing_x,Testing_y
    def data_divide(self,dataset):
        dataset = np.array(dataset)
        x = dataset[:,:-1]
        y = dataset[:,-1]
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
        #开始迭代
        error_0 = 0
        while counter <= MaxIteration:
            counter += 1
            error_1 = 0
            diff = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    diff[i] += (np.dot(theta,data_x[j]) - data_y[j]) * data_x[j,i]
                #print(diff[i])
            for a in range(n):
                theta[a] = theta[a] - (alpha * diff[a])/m
                #print("theta")
                #print(theta[i])
            print("theta:",theta)
            for k in range(m):
                error_1 += (data_y[k] - np.dot(theta,data_x[k]))**2 / (2*m)
            print("error:",error_1)
            if abs(error_1 - error_0) < epsilon:
                break
        return theta
        

    #利用梯度下降法得到的结果进行测试集数据的预测
    def Prediction(self,testing_x,theta):
        predcting_y = []
        for i in range(len(testing_x)):
            predcting_y.append(np.dot(testing_x[i],theta))
        print(predcting_y)
        return predcting_y
    
    #绘制图像
    def plot_data(self,predicting_data,real_data):
        #plt.figure(num = "Predicting_Data",figsize = (10,8),dpi = 120)
        x_value = list(range(0,len(predicting_data)))
        #plt.scatter()

test = Linear_regression
data_set = []
data_test = test.reading_data(test,'housing.data',data_set)
training_x,training_y,testing_x,testing_y = test.data_division(test,data_set)
length = len(training_x[0])
theta_init = np.random.random(size = length)
alpha = 0.0005
MaxIteration_1 = 20
epsilon_1 = 0.0005
#print(theta_init)
theta_result = test.Gradient_Descent(test,training_x,training_y,theta_init,alpha,MaxIteration_1,epsilon_1)
print(theta_result)


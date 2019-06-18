# -*- encoding = utf-8 -*-
#author: Yuhang Guo
'''
Using the cs229 courses in Standford
Destination:Learning the machine learning algorithms
Try to rewrite the .m matlab codes to .py python codes
'''
# 3. Softmax regression

import sys
sys.path.append(r'H:\UESTC\大三（下）\模式识别与统计学习\assignment\Mine_own_codes_and_reports\Assignments\logistic_regression')
import logistic_regression
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.datasets import load_iris



class Softmax_regression:
    #读取数据
    def load_data(self,filepath):
        '''
        :param filepath: one list，Store four file paths
        :return: four array or matrix
        '''
        idx3_data_train = logistic_regression.Logistic_regression.decode_idx3(logistic_regression,filepath[0])
        idx3_data_test = logistic_regression.Logistic_regression.decode_idx3(logistic_regression,filepath[1])
        idx1_data_train = logistic_regression.Logistic_regression.decode_idx1(logistic_regression,filepath[2])
        idx1_data_test = logistic_regression.Logistic_regression.decode_idx1(logistic_regression,filepath[3])
        return idx3_data_train,idx1_data_train,idx3_data_test,idx1_data_test

    #rebuild matrix
    def rebulit_data(self,idx3_data):
        shape_number = idx3_data.shape
        #print(shape_number)
        length = shape_number[0]
        width = shape_number[1] * shape_number[2]
        data_rebuilt = np.empty([length,width])
        for k in range(length):
            data_rebuilt[k] = np.ravel(idx3_data[k])
        data_one = np.ones(length)
        #print(np.c_[data_rebuilt,data_one])
        return np.c_[data_rebuilt,data_one]
    #define bool matrix
    def bool_function(self,y,label):
        return y == label
    def main_softmax_gradient(self,data_x,data_y,theta,alpha,lambda_regular):
        '''
        :param data_x:60000 * 785
        :param data_y: 60000 * 1
        :param theta: 10 * 785
        :param maxiteration: maximum iteration times
        :param alpha: learning rate
        :return:
        '''
        data_x = data_x
        #data_y = data_y + 1
        length,width = data_x.shape
        #print('data_dimension:',length,width)
        label_number = len(np.unique(data_y))
        label = np.arange(1,label_number+1)
        bool_matrix = label_binarize(data_y, classes=np.unique(data_y).tolist()).reshape(length,label_number)
        #print('bool_matrix dimension:',bool_matrix.shape,bool_matrix)# binary (m*k)
        #print('biaoqian shuliang',label_number)
        data_x_new = data_x.transpose() # n+1 * m
        #print('new data dimension:',data_x_new.shape)
        theta_data_x_new = theta.dot(data_x_new) # k * m
        theta_data_x_new = theta_data_x_new - np.max(theta_data_x_new)
        #print(np.exp(theta_data_x_new))
        possible_theta_data = np.exp(theta_data_x_new) / np.sum(np.exp(theta_data_x_new),axis=0)
        #print(possible_theta_data)
        cost_value = (-1 / length) * np.sum(np.multiply(bool_matrix,np.log(possible_theta_data).T)) + (lambda_regular / 2) * np.sum(np.square(theta))  #1 * 1
        gradient = (-1 / length) * (data_x_new.dot(bool_matrix - possible_theta_data.T)).T + lambda_regular * theta  # k * n+1
        return cost_value,gradient

    #train data
    def data_training(self,data_x,data_y,max_iteration,alpha,lambda_regular,epsilon):
        m,n = data_x.shape
        label_number = len(np.unique(data_y))
        #define one coefficients matrix，softmax: k kinds labels,every label has (width) coefficients
        theta = np.ones([label_number,n])
        print(theta)
        counter = 0
        error_0 = 0
        while counter < max_iteration:
            counter += 1
            error_1 = 0
            cost_value,gradient = Softmax_regression.main_softmax_gradient(self,data_x,data_y,theta,alpha,lambda_regular)
            print('Error value: %f' %(cost_value))
            #print(gradient)
            theta = theta - alpha * gradient
            error_1 = cost_value
            #print(theta)
            if abs(error_1 - error_0) < epsilon:
                break
            else:
                error_0 = error_1
            #print(error_0)
        return theta
    #testing data
    def predict(self,theta,testdata_x, testdata_y):  # testdata (10000 * 785)
        #testdata_y = testdata_y + 1
        prod = theta.dot(testdata_x.T)
        pred = np.exp(prod) / np.sum(np.exp(prod), axis=0)
        pred = pred.argmax(axis=0)
        accuracy = 0.0
        for i in range(len(testdata_y)):
            if testdata_y[i] == pred[i]:
                accuracy += 1
        return pred, float(accuracy / len(testdata_y))

if __name__ == '__main__':
    softmax = Softmax_regression
    filepath = ['train-images-idx3-ubyte','t10k-images-idx3-ubyte','train-labels-idx1-ubyte','t10k-labels-idx1-ubyte']
    data_trainX,data_trainY,data_testX,data_testY = softmax.load_data(softmax,filepath)
    #print(data_trainX.shape,data_trainY.shape,data_testX.shape,data_testY.shape)
    data_trainX = softmax.rebulit_data(softmax,data_trainX)
    data_trainX = data_trainX / 255.0
    data_testX = softmax.rebulit_data(softmax,data_testX)
    data_testX = data_testX / 255.0
    #print(data_trainX.shape,data_testX.shape)
    max_iteration = 1000
    alpha = 0.2
    lambda_regular = 0.001
    epsilon = 0.0001
    theta_result = softmax.data_training(softmax,data_trainX,data_trainY,max_iteration,alpha,lambda_regular,epsilon)
    print(theta_result)
    prediction,accuracy = softmax.predict(softmax,theta_result,data_testX,data_testY)
    print('Softmax_Regression Prediction Accuracy is percentage:%f.' %(accuracy * 100))




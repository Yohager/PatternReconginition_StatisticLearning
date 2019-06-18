# -*- encoding = utf-8 -*-
#author: Yuhang Guo
'''
Using the cs229 courses in Standford
Destination:Learning the machine learning algorithms
Try to rewrite the .m matlab codes to .py python codes
'''
# 2. Logistic regression

from sklearn.model_selection import train_test_split
import numpy as np 
import struct
import operator
from functools import reduce
from itertools import chain
import matplotlib.pyplot as plt

class Logistic_regression:
    #decode the idx3 files
    def decode_idx3(self,idx3_filepath):
        #read binary data
        binar_data = open(idx3_filepath,'rb').read()
        #magic num,dimension
        offset = 0
        fmt_header = ">iiii" # >:Big end storage i:Analytic type int
        magic_num, number_images,num_rows,num_cols = struct.unpack_from(fmt_header,binar_data,offset)
        #print("magic %d，number %d，size %d * %d" %(magic_num,number_images,num_rows,num_cols))
        #Parse to get the data set
        image_size = num_rows * num_cols
        offset += struct.calcsize(fmt_header)
        #print(offset)
        fmt_image = '>' + str(image_size) + 'B'
        #print(fmt_image,offset,struct.calcsize(fmt_image))
        images = np.empty((number_images,num_rows,num_cols))
        #plt.figure
        for i in range(number_images):
            if (i + 1) % 10000 == 0:
                print('Have parsed %d photos' %(i+1))
                #print(offset)
            images[i] = np.array(struct.unpack_from(fmt_image,binar_data,offset)).reshape((num_rows, num_cols))
            #print(images[i])
            offset += struct.calcsize(fmt_image)
        return images

    #decode the idx1 files
    def decode_idx1(self,idx1_filepath):
        #read binary data
        binar_data = open(idx1_filepath,'rb').read()
        #Parse header file
        offset = 0
        fmt_header = '>ii'
        magic_number, num_images = struct.unpack_from(fmt_header, binar_data, offset)
        #print('magic:%d, number: %d photos' % (magic_number, num_images))
        #Parse data set
        offset += struct.calcsize(fmt_header)
        fmt_image = '>B'
        labels = np.empty(num_images)
        for i in range(num_images):
            if (i + 1) % 10000 == 0:
                print('Have parsed %d photos' %(i + 1))
            labels[i] = struct.unpack_from(fmt_image,binar_data,offset)[0]
            offset += struct.calcsize(fmt_image)
        return labels

    #divide the data
    def data_divide(self,dataset):
        dataset = np.array(dataset)
        x = dataset[:,:-1]
        y = dataset[:,-1]
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
        data_one_1 = np.ones(len(x_train))
        data_one_2 = np.ones(len(x_test))
        x_train = np.c_[data_one_1,x_train]
        x_test = np.c_[data_one_2,x_test]
        #print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
        return x_train,x_test,y_train,y_test
    #built sigmoid function
    def sigmoid_function(self,z):
        return 1.0 / (1 + np.exp(-z))
    def max_likelihood_estimate(self,theta,x,y):
        z = theta * x
        h_x = Logistic_regression.sigmoid_function(self,z)
        likelihood_function = np.sum(y.T * np.log(h_x) + (1 - y).T * np.log(1 - h_x))
        return likelihood_function
    #logistic main part
    def main_logistic_gradient_ascent(self,training_data_x,training_data_y,max_iteration,alpha,epsilon):
        training_X = np.array(training_data_x)
        training_Y = np.array(training_data_y)
        print(training_Y.shape)
        m,n = training_X.shape
        counter = 0
        theta = np.ones(n)  #init
        #print(training_X.shape,theta.shape)
        error_0 = 0
        while counter < max_iteration:
            counter += 1
            error_1 = 0
            predict_Y = Logistic_regression.sigmoid_function(self, np.dot(training_X,theta))
            #print(predict_Y.shape)
            error = training_Y - predict_Y
            #print(error.shape)
            #calculate the gradient
            gradient = np.dot(training_X.T,error.T)
            #print(gradient.shape)
            theta = theta + alpha * gradient
            error_1 = np.sum(error)
            if abs(error_1 - error_0) < epsilon:
                break
            else:
                error_0 = error_1
            print(error_0)
        return error_0,theta,counter
    def predicting_data(self,theta, testing_data):
        row, col = testing_data.shape
        predict_label = np.zeros((row,1))
        for i in range(row):
            predict_label[i] = np.dot(theta,testing_data[i])
            if predict_label[i] >= 0.5:
                predict_label[i] = 1
            else:
                predict_label[i] = 0
        return predict_label
    def accurate_value(self,real_label,predict_label):
        total = len(real_label)
        counter = 0
        for i in range(total):
            if real_label[i] == predict_label[i]:
                counter += 1
        return counter / total

if __name__ == '__main__':
    Logistic = Logistic_regression
    binar_x = Logistic.decode_idx3(Logistic,'train-images-idx3-ubyte')
    shape_number = binar_x.shape  #three dimensions
    #print(shape_number[0],shape_number[1],shape_number[2])
    length = shape_number[0]
    width = shape_number[1] * shape_number[2]
    #print(width)
    data_resolved = np.empty([length,width])
    for k in range(length):
        data_resolved[k] = np.ravel(binar_x[k])  #change two dimension to one dimension as data_resolved
    #print(data_resolved.shape)
    binar_y = Logistic.decode_idx1(Logistic,'train-labels-idx1-ubyte')
    shape_number_y = binar_y.shape
    #print(shape_number_y)
    #print(binar_y)
    training_set = np.c_[data_resolved,binar_y]  #array joining
    #print(training_set.shape)
    training_set_new = training_set[np.where((binar_y == 0) | (binar_y == 1))] #we only need the '0' and '1' samples in minst
    #print(training_set_new.shape)
    x_train,x_test,y_train,y_test = Logistic.data_divide(Logistic,training_set_new)
    #print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
    alpha = 0.0001
    max_iteration = 1000
    epsilon = 0.0001
    error_result,theta,counter = Logistic.main_logistic_gradient_ascent(Logistic,x_train,y_train,max_iteration,alpha,epsilon)
    print('iteration number: %d'%(counter))
    print(theta)
    predict_label_result = Logistic.predicting_data(Logistic,theta,x_test)
    #print(predict_label_result)
    accuracy = Logistic.accurate_value(Logistic,y_test,predict_label_result)
    print('Logistic Regression Accuarcy value: %f'%(accuracy))







# -*- encoding = utf-8 -*-
#author: Yuhang Guo
'''
Using the cs229 courses in Standford
Destination:Learning the machine learning algorithms
Try to rewrite the .m matlab codes to .py python codes
'''
# 4. Principle Component Analysis

import numpy as np
from numpy.linalg import eig
import cv2
from PIL import Image
from matplotlib import pyplot as plt

class Principle_Component_Analysis:
    def load_data(self,filepath):
        input_image = Image.open(filepath)
        width = input_image.size[0]
        height = input_image.size[1]
        print(input_image)
        print('Picture size is %d * %d.' %(height,width))
        #input_image.show()
        image_pixel = np.array(input_image)
        return image_pixel

    def pre_processing(self,pixel_matrix):
        #decentration
        length,width = pixel_matrix.shape
        new_data = np.zeros([length,width])
        average = np.array(pixel_matrix.mean(axis = 0))
        #print(average)
        #print(pixel_matrix[0] - average)
        for i in range(len(pixel_matrix)):
            new_data[i] = pixel_matrix[i] - average
        #normalize
        #pixel_matrix = pixel_matrix / 255.0
        new_data = new_data / 255.0
        #print(new_data)
        return average,new_data

    def PCA_main_function(self,pixel_matrix,k):
        corr_matrix = np.dot(pixel_matrix,pixel_matrix.T)
        #print(corr_matrix.shape)
        eigenvalue,eigenvector = eig(corr_matrix)
        #print('eigenvalues: \n{}'.format(eigenvalue))
        #print('eigenvectors: \n{}'.format(eigenvector))
        #find out top-k eigenvalue and eigenvector
        sorted_indices = np.argsort(eigenvalue)
        top_k_eigenvalues = eigenvalue[sorted_indices[:-k-1:-1]]
        #print(top_k_eigenvalues)
        #print(sorted_indices[:-k-1:-1])
        top_k_eigenvector = eigenvector[:,sorted_indices[:-k-1:-1]]
        print(top_k_eigenvector.shape)
        pca_processing_data = np.dot(corr_matrix,top_k_eigenvector)
        print(pca_processing_data.shape)
        return pca_processing_data,eigenvalue

    #plot line chart
    def plot_line_chart(self,data):
        x_lab = list(range(len(data)))
        plt.plot(x_lab,data,label = 'Eigenvalues',color='b')
        plt.xlabel('Eigenvalues Index')
        plt.ylabel('Eigenvalues')
        plt.title('Sorted Eigenvalues Line Chart')
        plt.legend()
        plt.show()

    def MatrixToImage(self,data):
        new_im = Image.fromarray(data.astype(np.uint8))
        return new_im

if __name__ == '__main__':
    pca_test = Principle_Component_Analysis
    file_path = 'butterfly.bmp'
    pixel_matrix = pca_test.load_data(pca_test,file_path)
    length,width,height = pixel_matrix.shape
    print(length,width,height)
    red_pixel = np.array(pixel_matrix[:,:,0])
    green_pixel = np.array(pixel_matrix[:,:,1])
    blue_pixel = np.array(pixel_matrix[:,:,2])
    print(red_pixel)
    print(red_pixel.shape)
    #data pre-processing handle RGB
    average_1,red_pixel_preprocessing = pca_test.pre_processing(pca_test,red_pixel)
    average_2,green_pixel_preprocessing = pca_test.pre_processing(pca_test,green_pixel)
    average_3,blue_pixel_preprocessing = pca_test.pre_processing(pca_test,blue_pixel)
    #giving the dimensions number after reduction
    dimension_reduction_k = 50
    result_data_red,eigenvalues_red = pca_test.PCA_main_function(pca_test,red_pixel_preprocessing,dimension_reduction_k)
    result_data_green,eigenvalues_green = pca_test.PCA_main_function(pca_test,green_pixel_preprocessing,dimension_reduction_k)
    result_data_blue,eigenvalues_blue = pca_test.PCA_main_function(pca_test,blue_pixel_preprocessing,dimension_reduction_k)
    result_rgb = np.array([result_data_red.T,result_data_green.T,result_data_blue.T])
    print(result_rgb.T.shape)
    pca_test.plot_line_chart(pca_test,eigenvalues_red)
    pca_test.plot_line_chart(pca_test,eigenvalues_green)
    pca_test.plot_line_chart(pca_test,eigenvalues_blue)
    #old_im = pca_test.MatrixToImage(pca_test,pixel_matrix)
    #old_im.show()
    #new_im = pca_test.MatrixToImage(pca_test,result_rgb.T)
    #new_im.show()
    #green_pixel = pixel_matrix[:,:,1]
    # x_lab = list(range(length))
    # plt.plot(x_lab,eigenvalues)
    # plt.show()




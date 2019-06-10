# -*- encoding = utf-8 -*-
#author: Yuhang Guo
'''
Using the cs229 courses in Standford
Destination:Learning the machine learning algorithms
Try to rewrite the .m matlab codes to .py python codes
'''
# 5. Singular Value Decomposition


import numpy as np
from numpy.linalg import eig
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import pylab


class Singular_value_decomposition:
    def load_data(self,filepath):
        # PCA we use PIL.Image and now we use cv2 to load data and gray
        image = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
        #print(image.shape)
        #print(type(image))
        #print(image)
        return image

    def data_preprocessing(self,pixel_matrix):
        pixel_matrix_1 = pixel_matrix / 255.0
        return pixel_matrix_1

    def SVD_main_function(self,pixel_matrix,k):
        data_1 = np.dot(pixel_matrix,pixel_matrix.T)
        data_2 = np.dot(pixel_matrix.T,pixel_matrix)
        eigenvalues_1,eigenvectors_1 = np.linalg.eigh(data_1)
        #eigenvalues_2,eigenvectors_2 = np.linalg.eigh(data_2)
        #eigenvectors_2.transpose()
        eig_sort_1 = np.argsort(eigenvalues_1)
        top_k_eigenvalues = eigenvalues_1[eig_sort_1[:-k - 1:-1]]
        #eig_sort_2 = np.argsort(eigenvalues_2)
        top_k_eigenvector_1 = np.array(eigenvectors_1[:,eig_sort_1[:-k-1:-1]])
        print(top_k_eigenvector_1.shape)
        #top_k_eigenvector_2 = np.array(eigenvectors_2[:,eig_sort_2[:-k-1:-1]]).real   #V
        singular_values = np.sqrt(top_k_eigenvalues)  # sigma
        singular_matrix = np.mat(np.diag(singular_values))
        singular_inverse = singular_matrix.I   #V' = s^{-1} U.T A
        singular_v = singular_inverse.dot(top_k_eigenvector_1.T.dot(pixel_matrix))
        print(singular_v.shape)
        print(singular_matrix.shape)
        result_Matrix = np.dot((np.dot(top_k_eigenvector_1,singular_matrix)),singular_v)
        print(result_Matrix.shape)
        return result_Matrix

    def reconstruct_figure(self,init_data,pixel_matrix):
        restored_figure = pixel_matrix * 255.0
        #restored_figure_1 = restored_figure.real
        print(restored_figure)
        image_init = Image.fromarray(init_data)
        image_restored = Image.fromarray(restored_figure)
        fig = plt.figure()
        fig.suptitle('1')
        fig_1 = fig.add_subplot(1,2,1)
        fig_2 = fig.add_subplot(1,2,2)
        fig_1.imshow(init_data)
        fig_2.imshow(restored_figure)
        fig.show()
        pylab.show()




if __name__ == '__main__':
    Singular_vd = Singular_value_decomposition
    image_pixel = Singular_vd.load_data(Singular_vd,'Butterfly.bmp')
    image_pixel_preprocessing = Singular_vd.data_preprocessing(Singular_vd,image_pixel)
    print(image_pixel_preprocessing)
    result = Singular_vd.SVD_main_function(Singular_vd,image_pixel_preprocessing,50)
    #Singular_vd.reconstruct_figure(Singular_vd,image_pixel,result)
    u,sigma,v = np.linalg.svd(image_pixel_preprocessing)
    print(u.shape,sigma.shape,v.shape)
    u_1 = np.array(u[:,:50])
    sigma_1 = np.diag(sigma[:50])
    v_1 = v[:50]
    result_1 = np.dot(u_1,np.dot(sigma_1,v_1))
    Singular_vd.reconstruct_figure(Singular_vd,image_pixel,result)
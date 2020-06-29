#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 19:59:23 2020

@author: prarthanabhat
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# Download a video from youtube using : https://www.y2mate.com/en19

# Use ffmpeg to get frames -
# https://en.wikibooks.org/wiki/FFMPEG_An_Intermediate_Guide/image_sequence
##  ffmpeg -i hasReHalkat.mp4 ./input/image-%03d.png
# To create videos
## ffmpeg -i ./output/image-%03d.png output.mpg


# os.chdir('/Users/prarthanabhat/AI/2020/opencv_course/ExamplePost/Cartoon/')

def pencilSketch(image, arguments=0.02):
    
    ### YOUR CODE HERE
    kernelSize = 3
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Applying laplacian
    img1 = cv2.GaussianBlur(image,(3,3),0,0)
    laplacian = cv2.Laplacian(img1, cv2.CV_32F, ksize = kernelSize, 
                                scale = 1, delta = 0)
    # Normalize results
    cv2.normalize(laplacian, 
                    dst = laplacian, 
                    alpha = 0, 
                    beta = 1, 
                    norm_type = cv2.NORM_MINMAX, 
                    dtype = cv2.CV_32F)
    
    # Inverse Binary Thresholding
    th, dst_bin_inv = cv2.threshold(laplacian, laplacian.mean()+ arguments, 1, cv2.THRESH_BINARY_INV)
    
    pencilSketchImage = dst_bin_inv * 255
    pencilSketchImage = np.uint8(pencilSketchImage)
    pencilSketchImage = cv2.cvtColor(pencilSketchImage, cv2.COLOR_GRAY2BGR)

    return pencilSketchImage

def cartoonify(image, arguments=0.02):
    
    ### YOUR CODE HERE
    kSize = (5,5)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kSize)

    image = cv2.dilate(image, kernel1)
    pencilSketchImage = pencilSketch(image, arguments)
    cartoonImage = cv2.bitwise_and(image, pencilSketchImage)
    cartoonImage = cv2.erode(cartoonImage, kernel1)

    return cartoonImage

srcFiles = os.listdir('./input/')

for i in range(len(srcFiles)):
    if srcFiles[i].split('.')[1] == 'png':
        #print(srcFiles[i])
        imagePath = './input/' + srcFiles[i]
        image = cv2.imread(imagePath)
        #print('start2')
        cartoonImage = cartoonify(image, 0.03)
        #print('start3')
        pencilSketchImage = pencilSketch(image, 0.02)
        #print('start4')
        cv2.imwrite('./cartoon/'+ srcFiles[i], cartoonImage)
        cv2.imwrite('./pencil/'+ srcFiles[i], pencilSketchImage)

    '''
    # save
    plt.figure(figsize=[20,10])
    plt.subplot(131);plt.imshow(image[:,:,::-1]);
    plt.subplot(133);plt.imshow(cartoonImage[:,:,::-1]);
    plt.subplot(132);plt.imshow(pencilSketchImage[:,:,::-1]);
    plt.margins(0,0)
    plt.savefig('./output/'+ srcFiles[i])
    plt.close('all') '''
    
    



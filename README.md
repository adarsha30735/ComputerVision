# Face Detection, Edge Detection, and MAP Estimation

This repository contains implementations for face detection, edge detection, and Maximum A Posteriori (MAP) estimation. The project aims to showcase various image processing techniques using OpenCV and other relevant libraries.

## Table of Contents
- [Introduction](#introduction)
- [Script Code](#script-code)
  - [Face Detection](#face-detection)
  - [Edge Detection](#edge-detection)
  - [MAP Estimation](#map-estimation)
- [Usage](#usage)

- 

## Example

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import cv2 #only used to read the image


def canny_edge_detecor(sigma, low_hreshold, high_threshold, image, filter_x, filter_y):

    plt.imshow(image,cmap='gray')
    image = np.array(image)


#Gaussian Filter

    KERNEL_size = 3
    KERNEL = np.zeros((KERNEL_size, KERNEL_size))
    center = KERNEL_size // 2

    for i in range(KERNEL_size):
        for j in range(KERNEL_size):
            x = i - center
            y = j - center
            KERNEL[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            KERNEL_sum = np.sum(KERNEL)

# Normalize the KERNEL
    KERNEL /= KERNEL_sum

#Gaussian filter
    image = convolve(image,KERNEL)
    
#Perform the Convolution between filters and the image to obtain the gradient images
    Intensity_X= convolve(image, filter_x)
    Intensity_Y = convolve(image, filter_y)
    magnitude = np.sqrt(Intensity_X**2 + Intensity_Y**2)
    angle = np.arctan2(Intensity_Y, Intensity_X)

#Non-maximum Suppresion
    gradient_magnitude = magnitude
    gradient_direction = angle 

# Apply non-maximum suppression
    gradient_direction[gradient_direction < 0] += 180
    suppressed_image = np.zeros_like(gradient_magnitude)
    for i in range(1, suppressed_image.shape[0] - 1):
        for j in range(1, suppressed_image.shape[1] - 1):
            direction = gradient_direction[i, j]
            if (0 <= direction < 22.5) or (157.5 <= direction < 180):
                pixel_1 = gradient_magnitude[i, j+1]
                pixel_2 = gradient_magnitude[i, j-1]
            elif (22.5 <= direction < 67.5):
                pixel_1 = gradient_magnitude[i+1, j-1]
                pixel_2 = gradient_magnitude[i-1, j+1]
            elif (67.5 <= direction < 112.5):
                pixel_1 = gradient_magnitude[i+1, j]
                pixel_2 = gradient_magnitude[i-1, j]
            else:
                pixel_1 = gradient_magnitude[i-1, j-1]
                pixel_2 = gradient_magnitude[i+1, j+1]
            if gradient_magnitude[i, j] >= pixel_1 and gradient_magnitude[i, j] >= pixel_2:
                suppressed_image[i, j] = gradient_magnitude[i, j]
                               
# ApplyT1 and T2 thresholding
  
    weak_edge_pixels = (suppressed_image > low_threshold) & (suppressed_image <= high_threshold)
    strong_edge_pixels = suppressed_image > high_threshold
    strong_pixel_coordinates = np.argwhere(strong_edge_pixels)
    connected_pixels = set()
    for i, j in strong_pixel_coordinates:
        for ii in range(i-1, i+2):
            for jj in range(j-1, j+2):
                if (ii >= 0 and ii < image.shape[0] and jj >= 0 and jj < image.shape[1] and 
                    (ii, jj) != (i, j) and suppressed_image[ii, jj] > low_threshold):
                    connected_pixels.add((ii, jj))
    final_image = np.zeros_like(image)
    final_image[strong_edge_pixels] = 255
    for i, j in connected_pixels:
        final_image[i, j] = 255

    return final_image

def convolve(image, KERNEL):
    height, width = image.shape
    ksize = KERNEL.shape[0]
    padding_size = ksize // 2
    padded = np.pad(image, padding_size, mode='edge')
    convolved = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            convolved[i, j] = np.sum(padded[i:i+ksize, j:j+ksize] * KERNEL)
    return convolved



#define images

fish_boat = cv2.imread('fishingboat.tif', cv2.IMREAD_GRAYSCALE)
cameraman = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)
lena = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)


# Define the Sobel filters
filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
filter_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


# Define the Prewitt filters
prewitt_filter_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_filter_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

# Define the Robert filters
Robert_filter_x = np.array([[1, 0], [0, -1]])
Robert_filter_y = np.array([[0, -1], [1, 0]])


Fishing_boat_1 = canny_edge_detecor( 0.45,25,50, fish_boat,filter_x, filter_y)
Fishing_boat_2 = canny_edge_detecor( 0.1,10,20,fish_boat,filter_x, filter_y)
Fishing_boat_3 = canny_edge_detecor( 0.35,55,100,fish_boat,prewitt_filter_x, prewitt_filter_y)
Fishing_boat_4 = canny_edge_detecor( 0.95,15,30, fish_boat,Robert_filter_x, Robert_filter_y)

cv2.imwrite('fishingboat_result_1.png', Fishing_boat_1)
cv2.imwrite('fishingboat_result_2.png', Fishing_boat_2)
cv2.imwrite('fishingboat_result_3.png', Fishing_boat_3)
cv2.imwrite('fishingboat_result_4.png', Fishing_boat_4)

#Plot figures

print(' Edge detection of fishingboat picture using varying parameters ')
plt.subplot(411)
plt.imshow( Fishing_boat_1, cmap = 'gray')
plt.subplot(412)
plt.imshow(Fishing_boat_2, cmap = 'gray')
plt.subplot(413)
plt.imshow( Fishing_boat_3, cmap = 'gray')
plt.subplot(414)
plt.imshow( Fishing_boat_4, cmap = 'gray')
plt.show()

cameraman_1 = canny_edge_detecor( 0.75,55,110, cameraman,filter_x, filter_y)
cameraman_2 = canny_edge_detecor( 0.2,20,40,cameraman,filter_x, filter_y)
cameraman_3 = canny_edge_detecor( 0.5,45,90, cameraman,prewitt_filter_x, prewitt_filter_y)
cameraman_4 = canny_edge_detecor( 0.1,65,130, cameraman,Robert_filter_x, Robert_filter_y)

cv2.imwrite('cameraman_result_1.png', cameraman_1)
cv2.imwrite('cameraman_result_2.png', cameraman_2)
cv2.imwrite('cameramant_result_3.png', cameraman_3)
cv2.imwrite('cameraman_result_4.png', cameraman_4)

#Plot figures

print(' Edge detection of cameraman picture using varying parameters ')
plt.subplot(411)
plt.imshow( cameraman_1, cmap = 'gray')
plt.subplot(412)
plt.imshow(cameraman_2, cmap = 'gray')
plt.subplot(413)
plt.imshow( cameraman_3, cmap = 'gray')
plt.subplot(414)
plt.imshow( cameraman_4, cmap = 'gray')
plt.show()

lena = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
# Define the Sobel filters
filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
filter_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

lena_1 = canny_edge_detecor( 0.5,55,100, lena,filter_x, filter_y)
lena_2 = canny_edge_detecor( 0.35,25,45, lena,prewitt_filter_x, prewitt_filter_y)
lena_3 = canny_edge_detecor( 0.31,17,39,lena,filter_x, filter_y)
lena_4 = canny_edge_detecor( 0.3,27,50, lena,Robert_filter_x, Robert_filter_y)

cv2.imwrite('lena_1.png', lena_1)
cv2.imwrite('lena_2.png', lena_2)
cv2.imwrite('lena_3.png', lena_3)
cv2.imwrite('lena_4.png', lena_4)

#Plot figures

print(' Edge detection of lena picture using varying parameters ')
plt.subplot(411)
plt.imshow( lena_1, cmap = 'gray')
plt.subplot(412)
plt.imshow(lena_2, cmap = 'gray')
plt.subplot(413)
plt.imshow( lena_3, cmap = 'gray')
plt.subplot(414)
plt.imshow( lena_4, cmap = 'gray')
plt.show()

indoor = cv2.imread('indoor.tiff', cv2.IMREAD_GRAYSCALE)

indoor_1 = canny_edge_detecor( 0.45,25,50, indoor,filter_x, filter_y)
indoor_2 = canny_edge_detecor( 0.1,10,20,indoor,filter_x, filter_y)
indoor_3 = canny_edge_detecor( 0.35,55,100, indoor,prewitt_filter_x, prewitt_filter_y)
indoor_4 = canny_edge_detecor( 0.95,15,30, indoor,Robert_filter_x, Robert_filter_y)

cv2.imwrite('indoor_1.png', indoor_1)
cv2.imwrite('indoor_2.png', indoor_2)
cv2.imwrite('indoor_3.png', indoor_3)
cv2.imwrite('indoor_4.png', indoor_4)

#Plot figures

print(' Edge detection of indoor picture using varying parameters ')
plt.subplot(411)
plt.imshow( indoor_1, cmap = 'gray')
plt.subplot(412)
plt.imshow(indoor_2, cmap = 'gray')
plt.subplot(413)
plt.imshow( indoor_3, cmap = 'gray')
plt.subplot(414)
plt.imshow( indoor_4, cmap = 'gray')
plt.show()

#!/usr/bin/env python
# coding: utf-8

# In[88]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


import numpy as np

def buildRtable(reference_edges):
    num_angles = 8
    angle_division = 360 / num_angles
    height, width = reference_edges.shape
    center_x, center_y = width // 2, height // 2
    possible_reference_points = np.argwhere(reference_edges != 0)
    dists = np.sqrt(np.sum(( possible_reference_points - np.array([center_y, center_x]))**2, axis=1))
    ref_index = np.argmin(dists)
    ref_point = possible_reference_points[ref_index]
    r_table = {}
    for i, point in enumerate( possible_reference_points ):
        for j in range(num_angles):
            angle = j * angle_division
            r = round((point[1] - ref_point[1]) * np.cos(np.deg2rad(angle)) + (point[0] - ref_point[0]) * np.sin(np.deg2rad(angle)))
            if r not in r_table:
                r_table[r] = {}
            if j not in r_table[r]:
                r_table[r][j] = []
            r_table[r][j].append(point)
    return r_table



def genAccumulator(test_edges,r_table):

    sigma = 10
    num_angles = 8
    angle_division = 360 / num_angles
    accumulator = np.zeros(test_image.shape[:2], dtype=np.uint64)
    test_points = np.argwhere(test_edges != 0)
    num_test_points = len(test_points)
    for i in range(num_test_points):
        for j in range(num_angles):
            angle = j * angle_division
            r = round(test_points[i][1] * np.cos(np.deg2rad(angle)) + test_points[i][0] * np.sin(np.deg2rad(angle)))
            if r in r_table and j in r_table[r]:
                for reference_point in r_table[r][j]:
                    x, y = reference_point
                    dx = test_points[i][1] - x
                    dy = test_points[i][0] - y
                    weight = np.exp(-(dx**2 + dy**2)/(2*sigma**2))
                    displaced_x = round(x + dx)
                    displaced_y = round(y + dy)
                    if displaced_x < 0 or displaced_y < 0 or displaced_x >= test_image.shape[1] or displaced_y >= test_image.shape[0]:
                        continue
                    accumulator[displaced_y, displaced_x] += weight
    return accumulator


def find_maxima_location(accumulator):
    max_val = np.amax(accumulator)
    maxima_locations = np.argwhere(accumulator == max_val)
    if len(maxima_locations) > 1:
        center = np.array(accumulator.shape) / 2
        distances = np.linalg.norm(maxima_locations - center, axis=1)
        maxima_locations = np.array([maxima_locations[np.argmin(distances)]])
    return maxima_locations



def draw_rectangles_on_image(image, maxima_locations, size):
    output_image = image.copy()
    for location in maxima_locations:
        x, y = location[::-1] 
        cv2.rectangle(output_image, (x-size, y-size), (x+size, y+size), (0, 255, 255), 2)
        cv2.circle(output_image, (x, y), 5, (0, 0, 255), -1)
    return output_image


# In[89]:


# Load the reference image
ref_image_1 = cv2.imread('ref_img001.png', cv2.IMREAD_GRAYSCALE)
ref_image_2 = cv2.imread('ref_img002.png', cv2.IMREAD_GRAYSCALE)
ref_image_3 = cv2.imread('ref_img003.png', cv2.IMREAD_GRAYSCALE)

# Detect edges using the Canny edge detector
reference_edges_1 = cv2.Canny(ref_image_1, 50, 100)
reference_edges_2 = cv2.Canny(ref_image_2, 45, 80)
reference_edges_3 = cv2.Canny(ref_image_3, 60, 105)


# In[90]:


test_image_1 = cv2.imread('test_img001.png', cv2.IMREAD_GRAYSCALE)
test_image_2 = cv2.imread('test_img002.png', cv2.IMREAD_GRAYSCALE)
test_image_3 = cv2.imread('test_img003.png', cv2.IMREAD_GRAYSCALE)

# Detect edges using the Canny edge detector
test_edges_1 = cv2.Canny(test_image_1, 40, 83)
test_edges_2 = cv2.Canny(test_image_2, 59, 67)
test_edges_3 = cv2.Canny(test_image_3, 61, 93)


# In[91]:


#R table with 3 reference images
r_table_1 = buildRtable(reference_edges_1)
r_table_2 = buildRtable(reference_edges_2)
r_table_3 = buildRtable(reference_edges_3)


#Accumulators for first image with three different reference images
accumulator_11 = genAccumulator(test_edges_1, r_table_1)
accumulator_12 = genAccumulator(test_edges_1, r_table_2)
accumulator_13 = genAccumulator(test_edges_1, r_table_3)

#Accumulators for second image with three different reference images
accumulator_21 = genAccumulator(test_edges_2, r_table_1)
accumulator_22 = genAccumulator(test_edges_2, r_table_2)
accumulator_23 = genAccumulator(test_edges_2, r_table_3)

#Accumulators for third image with three different reference images

accumulator_31 = genAccumulator(test_edges_3, r_table_1)
accumulator_32 = genAccumulator(test_edges_3, r_table_2)
accumulator_33 = genAccumulator(test_edges_3, r_table_3)

#Find maximum peaks from 3 accumulators for test image 1

maxima_locations_11 = find_maxima_location(accumulator_11)
maxima_locations_12 = find_maxima_location(accumulator_12)
maxima_locations_13 = find_maxima_location(accumulator_13)

#Find maximum peaks from 3 accumulators for test image 2

maxima_locations_21 = find_maxima_location(accumulator_21)
maxima_locations_22 = find_maxima_location(accumulator_22)
maxima_locations_23 = find_maxima_location(accumulator_23)

#Find maximum peaks from 3 accumulators for test image 3

maxima_locations_31 = find_maxima_location(accumulator_31)
maxima_locations_32 = find_maxima_location(accumulator_32)
maxima_locations_33 = find_maxima_location(accumulator_33)


# Place rectangle and circle on the face of the image by refering to the detected peaks for test image 1
image_with_rectangles_11 = draw_rectangles_on_image(test_image_1, maxima_locations_11, 50)
image_with_rectangles_12 = draw_rectangles_on_image(test_image_1, maxima_locations_12, 50)
image_with_rectangles_13 = draw_rectangles_on_image(test_image_1, maxima_locations_13, 50)

# Place rectangle and circle on the face of the image by refering to the detected peaks for test image 2
image_with_rectangles_21 = draw_rectangles_on_image(test_image_2, maxima_locations_21, 50)
image_with_rectangles_22 = draw_rectangles_on_image(test_image_2, maxima_locations_22, 50)
image_with_rectangles_23 = draw_rectangles_on_image(test_image_2, maxima_locations_23, 50)

# Place rectangle and circle on the face of the image by refering to the detected peaks for test image 3
image_with_rectangles_31 = draw_rectangles_on_image(test_image_3, maxima_locations_31, 50)
image_with_rectangles_32 = draw_rectangles_on_image(test_image_3, maxima_locations_32, 50)
image_with_rectangles_33 = draw_rectangles_on_image(test_image_3, maxima_locations_33, 50)


# In[92]:


# Create a figure and subplots
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

# Plot the images in the subplots
axs[0, 0].imshow(cv2.cvtColor(image_with_rectangles_11, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title("Detected Object 1 with R table 1")
axs[0, 1].imshow(cv2.cvtColor(image_with_rectangles_12, cv2.COLOR_BGR2RGB))
axs[0, 1].set_title("Detected Object 1 with R table 2")
axs[0, 2].imshow(cv2.cvtColor(image_with_rectangles_13, cv2.COLOR_BGR2RGB))
axs[0, 2].set_title("Detected Object 1 with R table 3")
axs[1, 0].imshow(cv2.cvtColor(image_with_rectangles_21, cv2.COLOR_BGR2RGB))
axs[1, 0].set_title("Detected Object 2 with R table 1")
axs[1, 1].imshow(cv2.cvtColor(image_with_rectangles_22, cv2.COLOR_BGR2RGB))
axs[1, 1].set_title("Detected Object 2 with R table 2")
axs[1, 2].imshow(cv2.cvtColor(image_with_rectangles_23, cv2.COLOR_BGR2RGB))
axs[1, 2].set_title("Detected Object 2 with R table 3")
axs[2, 0].imshow(cv2.cvtColor(image_with_rectangles_31, cv2.COLOR_BGR2RGB))
axs[2, 0].set_title("Detected Object 3 with R table 1")
axs[2, 1].imshow(cv2.cvtColor(image_with_rectangles_32, cv2.COLOR_BGR2RGB))
axs[2, 1].set_title("Detected Object 3 with R table 2")
axs[2, 2].imshow(cv2.cvtColor(image_with_rectangles_33, cv2.COLOR_BGR2RGB))
axs[2, 2].set_title("Detected Object 3 with R table 3")

# Adjust the spacing 
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Save the figure 
fig.savefig('detected_objects.jpg', dpi=300, quality=100)


# In[72]:


# Save the images as JPEG 
cv2.imwrite('image_with_rectangles_11.jpg', image_with_rectangles_11)
cv2.imwrite('image_with_rectangles_12.jpg', image_with_rectangles_12)
cv2.imwrite('image_with_rectangles_13.jpg', image_with_rectangles_13)
cv2.imwrite('image_with_rectangles_21.jpg', image_with_rectangles_21)
cv2.imwrite('image_with_rectangles_22.jpg', image_with_rectangles_22)
cv2.imwrite('image_with_rectangles_23.jpg', image_with_rectangles_23)
cv2.imwrite('image_with_rectangles_31.jpg', image_with_rectangles_31)
cv2.imwrite('image_with_rectangles_32.jpg', image_with_rectangles_32)
cv2.imwrite('image_with_rectangles_33.jpg', image_with_rectangles_33)


# In[95]:


test_image_4 = cv2.imread('addtional_test.jpg', cv2.IMREAD_GRAYSCALE)
test_edges_4 = cv2.Canny(test_image_1, 60, 93)

accumulator_4 = genAccumulator(test_edges_4, r_table_1)

maxima_locations_4 = find_maxima_location(accumulator_4)

image_with_rectangles_4 = draw_rectangles_on_image(test_image_4, maxima_locations_4, 50)


# In[96]:




# Plot the image
plt.imshow(image_with_rectangles_4)

# Save the image
plt.savefig('image_with_rectangles.jpg', dpi=300, bbox_inches='tight')


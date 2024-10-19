# Face Detection, Edge Detection, and MAP Estimation

This repository  contains implementations for face detection, edge detection, and Maximum A Posteriori (MAP) estimation. The project aims to showcase various image processing techniques using OpenCV and other relevant libraries.

## Table of Contents
- [Introduction](#introduction)
- [Script Code](#script-code)
  - [Face Detection](#face-detection)
  - [Edge Detection](#edge-detection)
  - [MAP Estimation](#map-estimation)
- [Usage](#usage)

## Introduction
This project implements various image processing functionalities, including face detection using Haar cascades, edge detection with the Canny method, and MAP estimation for image analysis. It utilizes Python libraries such as OpenCV, NumPy, and Matplotlib to demonstrate these techniques.

## Script Code

### Face Detection

The script uses a pre-trained Haar cascade classifier to detect faces in images.

#### Full Code:
```python
import cv2

# Load the Haar cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the image
image = cv2.imread('path/to/your/image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the result
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()



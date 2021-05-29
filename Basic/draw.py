import cv2 as cv
import numpy as np 

def rescaleImage(frame, scale = 2.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions)

# Black (Blank) Image
blank = np.zeros((1000, 1000, 3), dtype = 'uint8')
# cv.imshow('Blank', blank)

# Pain an image
blank[:] = 255, 0, 0
cv.imshow("Blue", blank)

cv.imshow('Blue', blank)

# References range of pixels
blank[200:1200, 300:1400] = 0, 0, 255
cv.imshow("Blue + Red", blank)

# Draw a rectangle
image = np.zeros((1000, 1000, 3), dtype = "uint8")
cv.rectangle(image, (0, 0), (250, 250), (0, 255, 0), thickness = cv.FILLED)
cv.imshow('Rectangle', image)

image2 = np.zeros((1000, 1000, 3), dtype = "uint8")
cv.rectangle(image2, (0, 0), (image2.shape[1] // 2, image2.shape[0] // 2), (0, 255, 255), thickness = cv.FILLED)
cv.imshow('Rectangle', image2)

image3 = np.zeros((1000, 1000, 3), dtype = "uint8")
cv.circle(image3, (image3.shape[1] // 2, image3.shape[0] // 2), 500, (255, 0, 0), thickness = -1)
cv.imshow('Circle', image3)

image4 = np.zeros((1000, 1000, 3), dtype = "uint8")
cv.line(image4, (0, 0), (500, 500), (255, 255, 255), thickness = 5)
cv.imshow('Line', image4)

""" capture = cv.imread('Photos/dog.jpeg')
image_resize = rescaleImage(capture, scale = 3.0)
cv.imshow('Dog', image_resize) """ 

cv.waitKey(0)
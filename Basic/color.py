import cv2 as cv 
import matplotlib.pyplot as plt 

image = cv.imread('Photos/cat.jpeg')
cv.imshow('Cat', image)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Cat', gray)

hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
cv.imshow('HSV Cat', hsv) 

lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
cv.imshow('Lab Cat', lab)

lab_bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
cv.imshow('LAB_BGR', lab_bgr)

rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
cv.imshow('RGB Cat', rgb)

plt.imshow(image)
plt.show()

cv.waitKey(0)
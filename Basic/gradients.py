import cv2 as cv  
import numpy as np 

image = cv.imread("Photos/cat.jpeg")

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Laplacian 
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow("Laplacian", lap)

# Sobel 
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)

combined = cv.bitwise_or(sobelx, sobely)

cv.imshow("Horizontal", sobelx)
cv.imshow("Vertical", sobely)
cv.imshow("Combined", combined)

cv.waitKey(0)
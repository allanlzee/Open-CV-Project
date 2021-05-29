import cv2 as cv 

image = cv.imread('Photos/cat.jpeg')
cv.imshow("Cat", image)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)

threshold, thresh = cv.threshold(gray, 150, 225, cv.THRESH_BINARY)
cv.imshow("Threshold", thresh)

threshold2, thresh2 = cv.threshold(gray, 120, 225, cv.THRESH_BINARY)
cv.imshow("Thresh_Light", thresh2)

threshold_inv, thresh_inv = cv.threshold(gray, 120, 225, cv.THRESH_BINARY_INV)
cv.imshow("Thresh_INV", thresh_inv)

adaptive = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, 
cv.THRESH_BINARY, 11, 3)
cv.imshow("Adaptive", adaptive)

cv.waitKey(0)
import cv2 as cv
import numpy as np 

image = cv.imread('Photos/cat.jpeg')

#cv.imshow('Cats', image)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)

blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

resized = cv.resize(image, (image.shape[1] // 2, image.shape[0] // 2), interpolation = cv.INTER_AREA)
#cv.imshow("Cats Smaller", resized)

canny = cv.Canny(gray, 125, 175)
cv.imshow('Contour', canny)

ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow("Thresh", thresh)

contours, hierarchies = cv.findContours(canny, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found!')

blank = np.ones(image.shape, dtype='uint8')
cv.imshow("Blank", blank)

cv.drawContours(blank, contours, -1, (0, 0, 255), 2)
cv.imshow("Contours Drawn", blank)

cv.waitKey(0)
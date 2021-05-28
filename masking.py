import cv2 as cv 
import numpy as np 

image = cv.imread("Photos/cat.jpeg")

cv.imshow("Cats", image)

blank = np.zeros(image.shape[:2], dtype='uint8')
# cv.imshow("Blank", blank)

mask = cv.circle(blank, (image.shape[1]//2, image.shape[0]//2), 700, 255, -1)
cv.imshow("Mask", mask)

masked = cv.bitwise_and(image, image, mask=mask)
cv.imshow("Masked", masked)

cv.waitKey(0)

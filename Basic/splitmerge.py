import cv2 as cv 
import numpy as np

image = cv.imread("Photos/cat.jpeg")
cv.imshow("Cat Regular", image)

# Reconstruct image
b, g, r = cv.split(image)
k = np.zeros_like(b)

blank = np.ones(image.shape[:2], dtype='uint')
print(blank.shape)

blue = cv.merge([b, k, k])
green = cv.merge([k, g, k])
red = cv.merge([k, k, r])

cv.imshow("Blue", blue)
cv.imshow("Green", green)
cv.imshow("Red", red)

"""
# Split the channels
cv.imshow("Blue", b)
cv.imshow("Green", g)
cv.imshow("Red", r)

print(image.shape)
print(b.shape)
print(g.shape)
print(r.shape)

# Merge the channels
merged = cv.merge([b, g, r])
cv.imshow("Merged", merged)

"""


cv.waitKey(0)

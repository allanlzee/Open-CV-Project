import cv2 as cv
import numpy as np

# Translations 
def translate(image, x, y):
    transMatrice = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (image.shape[1], image.shape[0])
    return cv.warpAffine(image, transMatrice, dimensions)

image = cv.imread('Photos/dog.jpeg') 

cv.imshow("Dog Original", image)
translated = translate(image, -100, 100)
cv.imshow("Dog Translated", translated)

# Rotations
def rotate(image, angle, rotPoint = None):
    width = image.shape[1]
    height = image.shape[0]

    dimensions = (width, height)

    if rotPoint is None: 
        rotPoint = (width // 2, height // 2)

    rotMatrice = cv.getRotationMatrix2D(rotPoint, angle, 1.0)

    return cv.warpAffine(image, rotMatrice, dimensions)

rotated = rotate(image, 45)
cv.imshow("Dog Rotated", rotated)

rotated_rotated = rotate(rotated, -45)
cv.imshow("Rotated Rotated", rotated_rotated)

# Resizing
resized = cv.resize(image, (1000, 500), interpolation=cv.INTER_CUBIC)
cv.imshow("Resized Dog", resized)

flip = cv.flip(image, 0)
cv.imshow("Flipped Dog", flip)

flip2 = cv.flip(image, 1)
cv.imshow("Flipped Dog2", flip2)

flip3 = cv.flip(image, -1)
cv.imshow("Flipped Dog3", flip3)

cv.waitKey(0)
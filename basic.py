import cv2 as cv

image = cv.imread('Photos/dog.jpeg')
cv.imshow("Dog", image)

# Grayscaled Image
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("Grayscale", gray)

# Blurring Image
blur = cv.GaussianBlur(image, (3, 3), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

# Edge Cascades
cascade = cv.Canny(image, 125, 175)
cv.imshow("Edge Cascade", cascade)

cv.waitKey(0)

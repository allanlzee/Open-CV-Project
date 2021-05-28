import cv2 as cv 
import matplotlib.pyplot as plt
import numpy as np 

image = cv.imread("Photos/cat.jpeg")
cv.imshow("Cat", image) 

colors = ('b', 'g', 'r')

plt.figure()
plt.title("Colour Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

for i, col in enumerate(colors):
    hist = cv.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])

plt.show()

cv.waitKey(0)
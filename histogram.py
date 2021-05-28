import cv2 as cv
import matplotlib.pyplot as plt 

image = cv.imread("Photos/cat.jpeg")
cv.imshow("Cat", image)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("Gray-Scale", gray)

gray_hist = cv.calcHist([gray], [0], None, [256], [0, 256] )

plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(gray_hist)
plt.xlim([0, 256])
plt.show()

cv.waitKey(0)
import cv2 as cv 

image = cv.imread("Photos/cat.jpeg")
cv.imshow("Original", image)

average = cv.blur(image, (10, 10))
cv.imshow("Average", average)

gauss = cv.GaussianBlur(image, (5, 5), 0)
cv.imshow("Gauss", gauss)

median = cv.medianBlur(image, 3)
cv.imshow("Median", median)

bilateral = cv.bilateralFilter(image, 10, 35, 35)
cv.imshow("Bilateral", bilateral)

cv.waitKey(0)
cv.destroyAllWindows()
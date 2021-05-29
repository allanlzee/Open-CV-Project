import cv2 as cv 

haar_cascade = cv.CascadeClassifier('haar_face.xml')

image = cv.imread("Photos/raptors2.jpeg")
resized = cv.resize(image, (image.shape[1] * 2, image.shape[0] * 2))

cv.imshow("Lowry", resized)

gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
cv.imshow("Gray Lowry", gray)

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.04, minNeighbors=4)

print(f'Number of faces found = {len(faces_rect)}')

for (x, y, w, h) in faces_rect:
    cv.rectangle(resized, (x,y), (x + w, y + h), (0, 255, 0), thickness = 2)

cv.imshow('Lowry Face', resized)

cv.waitKey(0)
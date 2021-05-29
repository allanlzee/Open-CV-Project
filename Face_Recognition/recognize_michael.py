import cv2 as cv 
import numpy as np

haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Identify People
people = ["Michael", "Jim", "Pam", "Dwight"]

# Load numpy files 
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy', allow_pickle=True)

# Instantiate Face Recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.xml')

# <--------------------------------------------------------------------------------------------------------------> 

# Image Michael

# Read image 
image_michael = cv.imread(r'/Users/anniezhou/Desktop/Open-CV-Project/Face_Train/Michael/michael1.jpeg')

# Convert to gray scale
gray_michael = cv.cvtColor(image_michael, cv.COLOR_BGR2GRAY)

# cv.imshow("image_michael", gray)

# detect face
faces_rect = haar_cascade.detectMultiScale(gray_michael, 1.5, 4)

# draw rectangle around face
for (x, y, w, h) in faces_rect: 
    region = gray_michael[y:y+h, x:x+h]

    label, confidence = face_recognizer.predict(region)
    print(f'Label = {label} with confidence {confidence}')

    cv.putText(image_michael, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0),
    thickness = 2)
    cv.rectangle(image_michael, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv.imshow("Detected Michael", image_michael)

# <--------------------------------------------------------------------------------------------------------------> 

# Image 1

image_michael1 = cv.imread(r'/Users/anniezhou/Desktop/Open-CV-Project/Face_Train/Michael/michael2.jpeg')

gray_michael1 = cv.cvtColor(image_michael1, cv.COLOR_BGR2GRAY)

faces_rect1 = haar_cascade.detectMultiScale(gray_michael1, 1.8, 4)

for (x, y, w, h) in faces_rect1:
    region = gray_michael1[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(region)
    print(f'Label = {label} with confidence {confidence}')

    cv.putText(image_michael1, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness = 2)
    cv.rectangle(image_michael1, (x, y), (x + w, y + h), (0, 255, 0), thickness = 2)

cv.imshow("Detected Michael 1", image_michael1)

# <-------------------------------------------------------------------------------------------------------------->

# Image 2

image_michael2 = cv.imread(r'/Users/anniezhou/Desktop/Open-CV-Project/Face_Train/Michael/michael3.jpeg')

gray_michael2 = cv.cvtColor(image_michael2, cv.COLOR_BGR2GRAY)

faces_rect2 = haar_cascade.detectMultiScale(gray_michael2, 1.05, 4)

for (x, y, w, h) in faces_rect2:
    region = gray_michael2[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(region)

    cv.putText(image_michael2, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness = 2)
    cv.rectangle(image_michael2, (x, y), (x + w, y + h), (0, 255, 0), thickness = 2)

cv.imshow("Detected Michael 2", image_michael2)
# <--------------------------------------------------------------------------------------------------------------> 

# Image 3 

image_michael3 = cv.imread(r'/Users/anniezhou/Desktop/Open-CV-Project/Face_Train/Michael/michael4.jpeg')

gray_michael3 = cv.cvtColor(image_michael3, cv.COLOR_BGR2GRAY)

faces_rect3 = haar_cascade.detectMultiScale(gray_michael3, 1.8, 4)

for (x, y, w, h) in faces_rect3:
    region = gray_michael3[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(region)

    cv.putText(image_michael3, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness = 2)
    cv.rectangle(image_michael3, (x, y), (x + w, y + h), (0, 255, 0), thickness = 2)

cv.imshow("Detected Michael 3", image_michael3)

cv.waitKey(0) 
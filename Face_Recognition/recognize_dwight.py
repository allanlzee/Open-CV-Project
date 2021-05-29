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

# Image Dwight

# Read image 
image_dwight = cv.imread(r'/Users/anniezhou/Desktop/Open-CV-Project/Face_Train/Dwight/dwight1.jpeg')

# Convert to gray scale
gray_dwight = cv.cvtColor(image_dwight, cv.COLOR_BGR2GRAY)

# cv.imshow("Dwight", gray)

# detect face
faces_rect = haar_cascade.detectMultiScale(gray_dwight, 1.1, 4)

# draw rectangle around face
for (x, y, w, h) in faces_rect: 
    region = gray_dwight[y:y+h, x:x+h]

    label, confidence = face_recognizer.predict(region)
    print(f'Label = {label} with confidence {confidence}')

    cv.putText(image_dwight, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0),
    thickness = 2)
    cv.rectangle(image_dwight, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv.imshow("Detected Dwight", image_dwight)

# <--------------------------------------------------------------------------------------------------------------> 

# Image 1

image_dwight1 = cv.imread(r'/Users/anniezhou/Desktop/Open-CV-Project/Face_Train/Dwight/dwight2.jpeg')

gray_dwight1 = cv.cvtColor(image_dwight1, cv.COLOR_BGR2GRAY)

faces_rect1 = haar_cascade.detectMultiScale(gray_dwight1, 1.1, 4)

for (x, y, w, h) in faces_rect1:
    region = gray_dwight1[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(region)
    print(f'Label = {label} with confidence {confidence}')

    cv.putText(image_dwight1, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness = 2)
    cv.rectangle(image_dwight1, (x, y), (x + w, y + h), (0, 255, 0), thickness = 2)

cv.imshow("Detected Dwight1", image_dwight1)

# -------------------------------------------------------------------------------------------------------------->

# Image 2

image_dwight2 = cv.imread(r'/Users/anniezhou/Desktop/Open-CV-Project/Face_Train/Dwight/dwight3.jpeg')

gray_dwight2 = cv.cvtColor(image_dwight2, cv.COLOR_BGR2GRAY)

faces_rect2 = haar_cascade.detectMultiScale(gray_dwight2, 1.1, 4)

for (x, y, w, h) in faces_rect2:
    region = gray_dwight2[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(region)

    cv.putText(image_dwight2, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness = 2)
    cv.rectangle(image_dwight2, (x, y), (x + w, y + h), (0, 255, 0), thickness = 2)

cv.imshow("Detected Dwight2", image_dwight2)
# <--------------------------------------------------------------------------------------------------------------> 

# Image 3 

image_dwight3 = cv.imread(r'/Users/anniezhou/Desktop/Open-CV-Project/Face_Train/Dwight/dwight4.jpeg')

gray_dwight3 = cv.cvtColor(image_dwight3, cv.COLOR_BGR2GRAY)

faces_rect3 = haar_cascade.detectMultiScale(gray_dwight3, 1.1, 4)

for (x, y, w, h) in faces_rect3:
    region = gray_dwight3[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(region)

    cv.putText(image_dwight3, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness = 2)
    cv.rectangle(image_dwight3, (x, y), (x + w, y + h), (0, 255, 0), thickness = 2)

cv.imshow("Detected Dwight2", image_dwight3)

cv.waitKey(0) 
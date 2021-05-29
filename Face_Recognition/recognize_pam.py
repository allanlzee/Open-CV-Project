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

# Image Pam

# Read image 
image_pam = cv.imread(r'/Users/anniezhou/Desktop/Open-CV-Project/Face_Train/pam/pam1.jpeg')

# Convert to gray scale
gray_pam = cv.cvtColor(image_pam, cv.COLOR_BGR2GRAY)

# cv.imshow("Pam", gray)

# detect face
faces_rect = haar_cascade.detectMultiScale(gray_pam, 1.5, 4)

# draw rectangle around face
for (x, y, w, h) in faces_rect: 
    region = gray_pam[y:y+h, x:x+h]

    label, confidence = face_recognizer.predict(region)
    print(f'Label = {label} with confidence {confidence}')

    cv.putText(image_pam, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0),
    thickness = 2)
    cv.rectangle(image_pam, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv.imshow("Detected Pam", image_pam)

# <--------------------------------------------------------------------------------------------------------------> 

# Image 1

image_pam1 = cv.imread(r'/Users/anniezhou/Desktop/Open-CV-Project/Face_Train/pam/pam2.jpeg')

gray_pam1 = cv.cvtColor(image_pam1, cv.COLOR_BGR2GRAY)

faces_rect1 = haar_cascade.detectMultiScale(gray_pam1, 1.5, 4)

for (x, y, w, h) in faces_rect1:
    region = gray_pam1[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(region)
    print(f'Label = {label} with confidence {confidence}')

    cv.putText(image_pam1, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness = 2)
    cv.rectangle(image_pam1, (x, y), (x + w, y + h), (0, 255, 0), thickness = 2)

cv.imshow("Detected Pam 1", image_pam1)

# <-------------------------------------------------------------------------------------------------------------->

# Image 2

image_pam2 = cv.imread(r'/Users/anniezhou/Desktop/Open-CV-Project/Face_Train/pam/pam3.jpeg')

gray_pam2 = cv.cvtColor(image_pam2, cv.COLOR_BGR2GRAY)

faces_rect2 = haar_cascade.detectMultiScale(gray_pam2, 1.8, 4)

for (x, y, w, h) in faces_rect2:
    region = gray_pam2[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(region)

    cv.putText(image_pam2, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness = 2)
    cv.rectangle(image_pam2, (x, y), (x + w, y + h), (0, 255, 0), thickness = 2)

cv.imshow("Detected Pam 2", image_pam2)
# <--------------------------------------------------------------------------------------------------------------> 

# Image 3 

image_pam3 = cv.imread(r'/Users/anniezhou/Desktop/Open-CV-Project/Face_Train/pam/pam4.jpeg')

gray_pam3 = cv.cvtColor(image_pam3, cv.COLOR_BGR2GRAY)

faces_rect3 = haar_cascade.detectMultiScale(gray_pam3, 1.5, 4)

for (x, y, w, h) in faces_rect3:
    region = gray_pam3[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(region)

    cv.putText(image_pam3, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness = 2)
    cv.rectangle(image_pam3, (x, y), (x + w, y + h), (0, 255, 0), thickness = 2)

cv.imshow("Detected Pam 3", image_pam3)

cv.waitKey(0) 
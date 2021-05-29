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
image_jim = cv.imread(r'/Users/anniezhou/Desktop/Open-CV-Project/Face_Train/Jim/jim1.jpeg')

# Convert to gray scale
gray_jim = cv.cvtColor(image_jim, cv.COLOR_BGR2GRAY)

# cv.imshow("Dwight", gray)

# detect face
faces_rect = haar_cascade.detectMultiScale(gray_jim, 1.1, 4)

# draw rectangle around face
for (x, y, w, h) in faces_rect: 
    region = gray_jim[y:y+h, x:x+h]

    label, confidence = face_recognizer.predict(region)
    print(f'Label = {label} with confidence {confidence}')

    cv.putText(image_jim, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0),
    thickness = 2)
    cv.rectangle(image_jim, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv.imshow("Detected Jim", image_jim)

# <--------------------------------------------------------------------------------------------------------------> 

# Image 1

image_jim1 = cv.imread(r'/Users/anniezhou/Desktop/Open-CV-Project/Face_Train/Jim/jim2.jpeg')

gray_jim1 = cv.cvtColor(image_jim1, cv.COLOR_BGR2GRAY)

faces_rect1 = haar_cascade.detectMultiScale(gray_jim1, 1.1, 4)

for (x, y, w, h) in faces_rect1:
    region = gray_jim1[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(region)
    print(f'Label = {label} with confidence {confidence}')

    cv.putText(image_jim1, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness = 2)
    cv.rectangle(image_jim1, (x, y), (x + w, y + h), (0, 255, 0), thickness = 2)

cv.imshow("Detected Jim 1", image_jim1)

# -------------------------------------------------------------------------------------------------------------->

# Image 2

image_jim2 = cv.imread(r'/Users/anniezhou/Desktop/Open-CV-Project/Face_Train/Jim/jim3.jpeg')

gray_jim2 = cv.cvtColor(image_jim2, cv.COLOR_BGR2GRAY)

faces_rect2 = haar_cascade.detectMultiScale(gray_jim2, 1.5, 4)

for (x, y, w, h) in faces_rect2:
    region = gray_jim2[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(region)

    cv.putText(image_jim2, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness = 2)
    cv.rectangle(image_jim2, (x, y), (x + w, y + h), (0, 255, 0), thickness = 2)

cv.imshow("Detected Jim 2", image_jim2)
# <--------------------------------------------------------------------------------------------------------------> 

# Image 3 

image_jim3 = cv.imread(r'/Users/anniezhou/Desktop/Open-CV-Project/Face_Train/Jim/jim4.jpeg')

gray_jim3 = cv.cvtColor(image_jim3, cv.COLOR_BGR2GRAY)

faces_rect3 = haar_cascade.detectMultiScale(gray_jim3, 1.1, 4)

for (x, y, w, h) in faces_rect3:
    region = gray_jim3[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(region)

    cv.putText(image_jim3, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness = 2)
    cv.rectangle(image_jim3, (x, y), (x + w, y + h), (0, 255, 0), thickness = 2)

cv.imshow("Detected Jim 3", image_jim3)

cv.waitKey(0) 
import numpy as np
import cv2 as cv 

haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Identify People
people = ["Michael", "Jim", "Pam", "Dwight"]

# Load numpy files 
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy', allow_pickle=True)

# Instantiate Face Recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.xml')

# Dwight
# Read image 
image = cv.imread(r'/Users/anniezhou/Desktop/Open-CV-Project/Face_Train/Dwight/dwight1.jpeg')

# Convert to gray scale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# cv.imshow("Dwight", gray)

# detect face
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

# draw rectangle around face
for (x, y, w, h) in faces_rect: 
    region = gray[y:y+h, x:x+h]

    label, confidence = face_recognizer.predict(region)
    print(f'Label = {label} with confidence {confidence}')

    cv.putText(image, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0),
    thickness = 2)
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv.imshow("Detected Face", image)


# Michael

# Read in image
image_michael = cv.imread(r'/Users/anniezhou/Desktop/Open-CV-Project/Face_Train/Michael/michael1.jpeg')

# Convert to gray scale
gray_michael = cv.cvtColor(image_michael, cv.COLOR_BGR2GRAY)
# cv.imshow("Michael", gray_michael)

# Detect face
faces_rect_michael = haar_cascade.detectMultiScale(gray_michael, 1.1, 4)

# draw rectangle around face
for (x, y, w, h) in faces_rect_michael: 
    region = gray_michael[y:y+h, x:x+h]

    label, confidence = face_recognizer.predict(region)
    print(f'Label = {label} with confidence {confidence}')

    cv.putText(image_michael, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255),
    thickness = 2)
    cv.rectangle(image_michael, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)

cv.imshow("Detected Michael", image_michael)
cv.waitKey(0)
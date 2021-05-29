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

    cv.putText(image_michael, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0),
    thickness = 2)
    cv.rectangle(image_michael, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv.imshow("Detected Michael", image_michael)

# Pam
# Read image 
image_pam = cv.imread(r'/Users/anniezhou/Desktop/Open-CV-Project/Face_Train/Pam/pam1.jpeg')

# Convert to gray scale
gray = cv.cvtColor(image_pam, cv.COLOR_BGR2GRAY)

# cv.imshow("Pam", gray)

# detect face
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

# draw rectangle around face
for (x, y, w, h) in faces_rect: 
    region = gray[y:y+h, x:x+h]

    label, confidence = face_recognizer.predict(region)
    print(f'Label = {label} with confidence {confidence}')

    cv.putText(image_pam, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 0),
    thickness = 2)
    cv.rectangle(image_pam, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv.imshow("Detected Pam", image_pam)

# Jim
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

cv.imshow("Detected Face", image_jim)

cv.waitKey(0)
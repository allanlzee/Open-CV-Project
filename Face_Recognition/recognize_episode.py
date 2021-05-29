import cv2 as cv 
import numpy as np 

# Use haar_cascade to recognize faces
haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ["Angela", "Dwight", "Jim", "Michael", "Pam"]

# Load features from numpy
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy', allow_pickle=True)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.xml')

capture = cv.VideoCapture(0)

# Capture individual frames
while True:
    isTrue, frame = capture.read() 

    # frame acts as an image
    faces_rect = haar_cascade.detectMultiScale(frame, 1.8, 4)
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    for (x, y, w, h) in faces_rect:
        region = gray_frame[y:y+h, x:x+h]

        label, confidence = face_recognizer.predict(region)
        print(f'Label = {label} with confidence {confidence}')

        cv.putText(frame, str(people[label]), (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0),
        thickness = 2)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
    
    cv.imshow("Office", frame)
    
    if cv.waitKey(20) & 0xFF==ord('d'):
        break 

capture.release() 
cv.destroyAllWindows() 

cv.waitKey(0) 
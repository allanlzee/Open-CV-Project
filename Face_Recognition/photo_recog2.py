import cv2 as cv  
import os 
import numpy as np 

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ["Michael", "Jim", "Pam", "Dwight"]

DIR = r'/Users/anniezhou/Desktop/Open-CV-Project/Face_Train'

features = []
labels = [] 

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for image in os.listdir(path):
            image_path = os.path.join(path, image)

            image_array = cv.imread(image_path)
            gray = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, 
            minNeighbors = 4)

            for (x, y, w, h) in faces_rect: 
                region = gray[y:y+h, x:x+w]
                features.append(region)
                labels.append(label)

create_train()

# print(f'Length of the features = {len(features)}')
# print(f'Length of the labels = {len(labels)}')

print("Training <-----------------------------> Done")
features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create() 
face_recognizer.train(features, labels)

face_recognizer.save('face_trained.xml')
np.save("features.npy", features)
np.save("labels.npy", labels)
import cv2 as cv 

haar_cascade = cv.CascadeClassifier('haar_face.xml')

capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()

    faces_rect = haar_cascade.detectMultiScale(frame, scaleFactor=1.6, minNeighbors=4)
    # print(f'Number of faces found = {len(faces_rect)}')

    for (x, y, w, h) in faces_rect:
        cv.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), thickness = 2)

    cv.imshow("Video", frame)
    
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destoryAllWindows()
cv.waitKey(0)
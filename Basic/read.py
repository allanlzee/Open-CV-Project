import cv2 as cv

print('OpenCV version {0}'.format(cv.__version__))

image = cv.imread('Photos/cat.jpeg')

cv.imshow("Cats", image)

# Use built in camera
capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)

    # Stop the video from playing forever
    # If the letter d is pressed, it will break out
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()

# Key Board Binding
cv.waitKey(0)
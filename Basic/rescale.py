import cv2 as cv

# Resize Function 
def rescaleFrame(frame, scale = 0.75):
    # Rescale the frame of the image/video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

""" image = cv.imread("Photos/dog.jpeg")
image_resize = rescaleFrame(image)
cv.imshow("Dog", image_resize)  """

# (0) -> Computer Camera
capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()

    resized = rescaleFrame(frame, scale = 0.75)
    cv.imshow("Video", frame)
    cv.imshow("Video Resized", resized)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

# Close all windows
capture.release()
cv.destroyAllWindows()

cv.waitKey(0)
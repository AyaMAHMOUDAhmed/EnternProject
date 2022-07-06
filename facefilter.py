import cv2
import numpy as np

# Define filters
filters = ['images/phar1.png', 'images/phar2.png']
expanders = [
    [0.9, 0.3, 1.3, 1.9],
    [0.8, 0.55, 1.55, 1.7]
]
filterIndex = 0

haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Load the video
camera = cv2.VideoCapture(0)

# Keep looping
while True:
    # Grab the current paintWindow
    (ret, frame) = camera.read()
    if ret:
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640,480))
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply detection algorithm to grayscale image
        faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

        # For each detected face, draw bounding box
        for (x, y, w, h) in faces:
            x, y, w, h = (np.array([x,y,w,h]) * np.array(expanders[filterIndex])).astype(int)
            x1, x2, y1, y2 = x, x+w, y, y+h
    #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            _filter = cv2.imread(filters[filterIndex], cv2.IMREAD_UNCHANGED)

            filter_width = int(w)
            filter_height = int(h)
            filter_resized = cv2.resize(_filter, (filter_width, filter_height), interpolation = cv2.INTER_CUBIC)
            frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * (1 - filter_resized[:, :, 3:] / 255) + \
                                  filter_resized[:, :, :3] * (filter_resized[:, :, 3:] / 255)
        cv2.imshow("Selfie Filters", frame)

    # If the 'q' key is pressed, stop the loop
    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        break
    elif key & 0xFF == ord("s"):
        cv2.imwrite("capture.jpg", frame)

# Cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

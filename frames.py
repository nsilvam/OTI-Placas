import os.path
import cv2

path = 'C:/Users/Usuario/Desktop/labsmart/converted'
directory = 'C:/Users/Usuario/Desktop/labsmart'

cont = 0
img = 1

for filename in os.listdir(directory):
    if filename.endswith(".avi"):
        print("Video: %s", filename)
        capture = cv2.VideoCapture(os.path.join(directory, filename))
        cont = 0
        # while True:
        while capture.isOpened():
            ret, frame = capture.read()
            if ret:
                # extract 1 image within 30 frames (1 second of video)
                if cont % 30 == 0:
                    cv2.imwrite(path + 'img_%04d.jpg' % img, frame)
                    img += 1
                cont += 1
            else:
                break

capture.release()
cv2.destroyAllWindows()

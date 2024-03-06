""" Paul Viola + Micheal Jones worked with together in 2001. Viola Jones algorithm is detecting faces on image and videos.
It works well on frontal faces. It uses grayscale images.It detects faces not identity.
Algorithm has 2 parts:
1.Training
2.Detecting
Haar-like features:
It measures and calculates intensity differences of rectangular areas.

Haarcascade:
It is a pre-trained classifier model used for object detection.
It belongs to OpenCV

OpenCV:
It has been using for image and video processing algorithms include object detection, face recognition, edge detection, image classification, stereo vision, camera calibration.

Dlib(Dahlia Library):
It has been using for face recognition, object detection, facial feature extraction, face recognition, gender prediction, emotional analysis.
It is available in Python and C++.
It can be used in RGB image

68 Facial Landmark Point:
It founds 68 different points in faces. Dlib's shape_predictor class is defines 68 Facial Landmark Point
"""


import dlib
import cv2
import matplotlib.pyplot as plt

detector = dlib.get_frontal_face_detector()

image = cv2.imread("images\\input\\miranda_face.jpg", cv2.IMREAD_COLOR)
print(image.shape)

# size = 512
# resized_image = cv2.resize(image, (size, size))


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = detector(gray)

for face in faces:
    left, top, width, height = face.left(), face.top(), face.width(), face.height()
    
    cv2.rectangle(image, (left, top), (left+width, top+height), (200, 200, 200), 2)

cv2.imshow("Detected Face", image)
cv2.imwrite("images\\output\\face_detection_dlib.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()




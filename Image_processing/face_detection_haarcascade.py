import cv2
import matplotlib.pyplot as plt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

image = cv2.imread("images/input/miranda_face.jpg", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 200, 200), 2)

# cv2.imshow("Face Detection", image)
# cv2.imwrite("images\\output\\face_detection_haarcascade.jpg", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
dlib_image = cv2.imread("images\\output\\face_detection_dlib.jpg")
haar_image = cv2.imread("images\\output\\face_detection_haarcascade.jpg")
point_image = cv2.imread("images\\output\\face_detection_points.jpg")
# image_rgb = cv2.imread(image, cv2.COLOR_BGR2RGB)
dlib_image_rgb = cv2.cvtColor(dlib_image, cv2.COLOR_BGR2RGB)
haar_image_rgb = cv2.cvtColor(haar_image, cv2.COLOR_BGR2RGB)
point_image_rgb = cv2.cvtColor(point_image, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))  

axes[0,0].imshow(image)
axes[0,0].set_title('Normal Image')

axes[0,1].imshow(dlib_image_rgb)
axes[0,1].set_title('Face Detected with Dlib')

axes[1,0].imshow(haar_image_rgb)
axes[1,0].set_title('Face Detected with Haarcascade')

axes[1,1].imshow(point_image_rgb)
axes[1,1].set_title('Face Detected with 68 Facial Landmark Point')

plt.show()
fig.savefig("images/output/all_detections.jpg")
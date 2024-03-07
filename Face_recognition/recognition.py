import cv2
import matplotlib.pyplot as plt
import face_recognition
import dlib

def face_recognition_photo_dlib(photo_path):
    """Dahlia lib ile face detection fonksiyonu"""
    image = cv2.imread(photo_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    return face_locations

def face_recognition_photo_haarcascade(photo_path):
    """Haarcascade ile face detection fonksiyonu"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(photo_path, cv2.IMREAD_COLOR )
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=3, minSize=(30,30))
    for (x, y, w, h) in faces:
        cv2.rectangle(image,(x,y), (x+w, y+h), (200,200,200),2)
    cv2.imshow("face detected in image with HAARCASCADE", image)
    cv2.imwrite("Face_recognition\\output\\haarcascadexxx.jpg",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image



def face_recognition_video(video_path):
    """opencv ile birlikte haarcascade ile videoda face detection fonksiyonu"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap =cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.imshow('Face Detection || press "x" to exit', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    
    # def face_recognition_in_gif(gif_path):
#     gif = cv2.VideoCapture(gif_path)
#     face_locations = []
#     while True:
#         ret, frame = gif.read()   
#         if not ret:
#             break
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         face_locations_in_frame = face_recognition.face_locations(rgb_frame)
#         face_locations.extend(face_locations_in_frame)
#     gif.release()
#     return face_locations

if __name__ == "__main__":
    #face_recognition_photo_haarcascade("Face_recognition/input/miranda_face.jpg")
    face_recognition_video("Face_recognition/input/head-pose-face-detection-female.mp4")    
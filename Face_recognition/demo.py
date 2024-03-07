from recognition import face_recognition_photo_dlib, face_recognition_video

photo_face_locations = face_recognition_photo_dlib("images\\input\\miranda_face.jpg")
video_face_locations = face_recognition_video("videos\\head-pose-face-detection-female.mp4")
# gif_face_locations = face_recognition_in_gif("animation.gif")

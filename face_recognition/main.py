# IMPORTS
import cv2 as cv
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# INITIALIZATION
facenet = FaceNet()

# Yuklangan embeddinglar va o'rgatilgan SVM modelini chaqiramiz
faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
Y = faces_embeddings['arr_1']

encoder = LabelEncoder()
encoder.fit(Y)

# Haarcascade yuklash (yuzni aniqlash uchun)
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

# SVM modelini yuklash
model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

# Kamera sozlamalari
cap = cv.VideoCapture(0)  # 0 -> asosiy kamera

# REAL VAQT YUZNI TANISH
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Kameradan tasvir olinmadi. Iltimos, kamerani tekshiring.")
        break

    # Tasvirni RGB va Gray formatga o'zgartirish
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Yuzni aniqlash
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)

    for x, y, w, h in faces:
        # Yuzni kesib olish va hajmini o'zgartirish
        face_img = rgb_img[y:y + h, x:x + w]
        face_img = cv.resize(face_img, (160, 160))  # FaceNet talabiga muvofiq hajm
        face_img = np.expand_dims(face_img, axis=0) / 255.0  # Normalizatsiya

        # FaceNet yordamida embedding olish
        embedding = facenet.embeddings(face_img)

        # SVM modeli yordamida yuzni tanish
        face_name = model.predict(embedding)
        final_name = encoder.inverse_transform(face_name)[0]

        # Ekranda aniqlangan yuz va nomni ko'rsatish
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv.putText(frame, str(final_name), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

    # Tasvirni ko'rsatish
    cv.imshow("Face Recognition", frame)

    # Q tugmasi bosilsa, dasturdan chiqish
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera va oynalarni yopish
cap.release()
cv.destroyAllWindows()

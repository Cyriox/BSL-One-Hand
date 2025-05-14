# File: run_app.py

import tensorflow as tf
import cv2
import numpy as np
import pyttsx3
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder


# Load model and encoder
model = load_model("models/bsl_model.h5")
encoder = LabelEncoder()
encoder.classes_ = np.load("models/label_encoder_classes.npy", allow_pickle=True)

# TTS
engine = pyttsx3.init()
def speak(text):
    engine.say(str(text))
    engine.runAndWait()

# MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    image = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        landmarks = landmarks[:63] + [0]*(63 - len(landmarks))

        prediction = model.predict(np.array([landmarks]), verbose=0)
        predicted_class = encoder.inverse_transform([np.argmax(prediction)])[0]

        cv2.putText(image, f'Sign: {predicted_class}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
        

    cv2.imshow("BSL to Speech", image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

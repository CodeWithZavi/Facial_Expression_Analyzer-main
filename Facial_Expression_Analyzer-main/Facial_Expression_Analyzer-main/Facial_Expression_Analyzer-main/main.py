import cv2
import numpy as np
from tensorflow.keras.models import load_model



print("RUNNING UPDATED FILE")
model = load_model("model/emotion_model.h5", compile=False)


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Webcam not reading")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face, (64, 64))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        print("FACE SHAPE:", face.shape)

        prediction = model.predict(face, verbose=0)
        emotion = emotion_labels[np.argmax(prediction)]

        cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame, emotion, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0),2)

    cv2.imshow("Facial Expression Analyzer", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

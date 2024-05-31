import cv2
import numpy as np
from keras.models import load_model
 
model = load_model("KerasModel.h5")
 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
 
labels_dict = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
 
video_capture = cv2.VideoCapture(0)
 
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        resized_face = cv2.resize(face_roi, (48, 48))
        
        reshaped_face = np.stack((resized_face,) * 3, axis=-1)
        reshaped_face = np.reshape(reshaped_face, (1, 48, 48, 3))
        result = model.predict(reshaped_face)
        label_index = np.argmax(result)
        emotion_label = labels_dict[label_index]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (48, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (48, 0, 0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
video_capture.release()
cv2.destroyAllWindows()
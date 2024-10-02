import cv2
import numpy as np
from tensorflow.keras.models import load_model
from deepface import DeepFace


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


model = load_model('gender_classification_model.h5')


def predict_gender(face_roi):

    face_roi_resized = cv2.resize(face_roi, (150, 150))
    
    face_roi_normalized = face_roi_resized / 255.0
    
    face_roi_reshaped = np.expand_dims(face_roi_normalized, axis=0)
    
    prediction = model.predict(face_roi_reshaped)
    
    gender = 'Feminino' if prediction[0][0] < 0.5 else 'Masculino'
    return gender

cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]

        try:
            gender = predict_gender(face_roi)

            emotion_result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            if isinstance(emotion_result, list):
                emotion_result = emotion_result[0]

            emotion = emotion_result.get('dominant_emotion', 'N/A')
            
            text = f"{emotion}, {gender}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        except Exception as e:
            print(f"Erro na análise: {str(e)}")

    cv2.imshow('Deteccao Avancada - Prototipo', frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

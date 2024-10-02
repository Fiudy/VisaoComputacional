import cv2
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.HandTrackingModule import HandDetector


import os



video = cv2.VideoCapture(0)

face_detector = FaceDetector()
hand_detector = HandDetector(detectionCon=0.5)  

while True:
    ret, img = video.read()

 
    if not ret:
        print("Falha ao ler imagem da câmera")
        break
    
    # img = cv2.flip(img, 1)

    img, bboxes = face_detector.findFaces(img, draw=True)

    
    if not isinstance(img, np.ndarray):
        print("Invalid image after face detection")
        continue

   
    hands, _ = hand_detector.findHands(img, draw=True) 
    
    
    img = cv2.resize(img, (970, 520)) 
    
    cv2.imshow("Resultado", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # o 'q' fecha a tela
        break


recognizer = cv2.face.LBPHFaceRecognizer_create()



faces, _ = face_detector.findFaces(img)
ids = [1, 2, 3]  
recognizer.train(faces, np.array(ids))


faces, _ = face_detector.findFaces(img)


for (x, y, w, h) in faces:
    
    face_region = img[y:y+h, x:x+w]

    
    id, _ = recognizer.predict(face_region)

    
    cv2.putText(img, "Person: " + str(id), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

video.release()
cv2.destroyAllWindows()
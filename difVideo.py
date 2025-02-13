import tensorflow as tf
from deepface import DeepFace
import cv2


reference_image = cv2.imread('/home/sepide/Downloads/meli.jpeg')

video_capture = cv2.VideoCapture('/home/sepide/Downloads/meli.mp4')
cv2.namedWindow('Video Frame capture',0)
while True:
    ret,frame = video_capture.read()

    try:
        result= DeepFace.verify(reference_image,frame, detector_backend='dlib', model_name='Dlib')

        face_box = result["facial_areas"]["img2"]
        x,y,w,h = face_box['x'], face_box['y'], face_box['w'], face_box['h']
        cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,0), 2)

        faceMatchingScore="{:.8f}".format(float(result["distance"]))
        font = cv2.FONT_HERSHEY_SIMPLEX
        org=(50,50)
        fontScale=int(1.0)
        color =(255,0,0)
        thickness=2

        # cv2.putText(frame, f'Score:{faceMatchingScore}', org, fontScale, color, thickness, cv2.LINE_4)

    except ValueError as e:
        print(f"no face detected : {e}")
        faceMatchingScore="N/A"

        cv2.imshow("vide", frame)

        if cv2.waitKey(100) :
            break

video_capture.release()
cv2.destroyAllWindows()

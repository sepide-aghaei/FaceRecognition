from deepface import DeepFace
import cv2

p1=cv2.imread('/home/sepide/Downloads/s.jpeg')
p2=cv2.imread('/home/sepide/Downloads/meli.jpeg')

result = DeepFace.verify(p1,p2,detector_backend='dlib', model_name='Dlib')

print(result)
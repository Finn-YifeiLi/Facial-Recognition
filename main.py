import cv2
import numpy
import dlib
import sys
import pickle

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()

with open('knownPeople.pkl', 'rb') as file:
    knownPeople = pickle.load(file)

def recognize_face(encoding, frame, x,y):
    for name, knownEncoding in knownPeople.items():
        cosine_theta = numpy.dot(encoding, knownEncoding) / numpy.linalg.norm(encoding) / numpy.linalg.norm(knownEncoding)
        angle = numpy.arccos(cosine_theta)/3.14*180
        
        threshold = 20
        if angle < threshold:
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0,0,255))
            return
    cv2.putText(frame, 'Unknown', (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0,0,255))
        

def live_recognize():
    web_cam = cv2.VideoCapture(0)
    while True:
        ret, frame = web_cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        # find face
        for face in faces:
            x,y,w,h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 10)

            # Recognize People
            landmarks = predictor(gray, face)
            encoding = face_encoder.compute_face_descriptor(frame, landmarks)
            recognize_face(numpy.array(encoding), frame, x, y)

        cv2.imshow("Web Cam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    web_cam.release()
    cv2.destroyAllWindows()
    
live_recognize()
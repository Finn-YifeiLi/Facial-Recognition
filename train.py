import cv2
import dlib
import numpy
import pickle
import os
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()

knownPeople = {}

# Train my laptop to know people
def remember(img, name):
    # getting gray image
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the coordinates of faces in gray
    faces = detector(gray)

    for face in faces:
        # Find landmarks and encode faces
        landmarks = predictor(gray, face)
        face_encoding = face_encoder.compute_face_descriptor(image, landmarks)
        knownPeople[name] = numpy.array(face_encoding)


for file_name in os.listdir("Faces"):
    if file_name == '.DS_Store':
        continue
    file_path = os.path.join("Faces", file_name)
    file_name = file_name.split(".")[0]
    remember(file_path, file_name)

with open('knownPeople.pkl', 'wb') as file:
    pickle.dump(knownPeople, file)
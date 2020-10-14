import face_recognition
import numpy as np


image = face_recognition.load_image_file("/home/giftiay/PycharmProjects/awesome-cool/unknown_pictures/YZ2C.jpg")
face_locations = face_recognition.face_locations(image)

know_code = np.load("/home/giftiay/PycharmProjects/awesome-cool/util/yzc_face_npy.npy")

face_recognition.compare_faces(know_code,face_locations)
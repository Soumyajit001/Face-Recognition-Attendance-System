import cv2
import numpy as np
import os
import pickle

video = cv2.VideoCapture(0)  # 0 is for webcam
face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  # OpenCV uses haar Cascade Classifier to detect the faces in the webcam
# Inside the bracket we are going to import the XML file. XML file for Frontal face detection. For that we are going to use the Google by searching haarcascade_frontalface_default.xml and download it from github. Copy the file from the download folder and paste it in our project folder.

face_data = []

i = 0

name = input("Enter your name: ")

while True:
    ret, frame = video.read()  # ret, frame are going to read web camera
    if not ret:  # Handle the case where the video stream ends or the frame cannot be read.
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # We are converting our colour ful images to black and white images because our cascade classifier uses gray scale images for detecting the faces.
    faces = face_detect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)         # Using the scaling functions

    for(x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w, :]
        resized_img = cv2.resize(crop_img, dsize=(50, 50))
        if len(face_data) <= 100 and i % 10 == 0:
            cv2.putText(frame, str(len(face_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)      # cv2.putText(frame, str(len(face_data)), org=(50, 50), cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(50, 50, 255), thickness=1)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

        cv2.imshow("frame", frame)      # cv2.imshow(winname="frame", frame)
        k = cv2.waitKey(1)
        if len(face_data) == 50:         # Exits on pressing the key with ASCII value 50 ('2')
            break

video.release()
cv2.destroyAllWindows()


# save faces in pickle file
face_data = np.array(face_data)     # converted the normal array to numpy array
face_data = face_data.reshape(100, -1)

# this pickle file for saving the students name for the attendance
if 'names.pkl' not in os.listdir('Data/'):      # If the pickle file is not present in our os we are going to create a new folder
    names = [name]*100       # A new list is created with 100 identical elements, all being the value of name. [name]: This creates a list containing a single element, which is the value of the variable name & *100: This multiplies the list by 100, effectively replicating the list's contents 100 times.
    with open('Data/names.pkl', 'wb') as f:     # wb: writing format
        pickle.dump(names, f)

# this function is used for loading all the names in the pickle file of our dataset
else:
    with open('Data/names.pkl', 'rb') as f:     # If the pickle file is present, we are going to update the values
        names = pickle.load(f)    # we are going to load the file for our dataset
    names = names + [name]*100

    # this function is used to update the values in the pickle file which we are going to use it
    with open('Data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

if 'face_data,pkl' not in os.listdir('Data/'):      # this pickle file for saving the faces detected while using the web camera
    with open('Data/face_data,pkl', 'wb') as f:
        pickle.dump(face_data, f)
else:
    with open('Data/face_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, face_data, axis=0)
    with open('Data/face_data,pkl', 'wb') as f:
        pickle.dump(faces, f)


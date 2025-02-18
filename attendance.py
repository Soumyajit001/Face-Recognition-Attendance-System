import cv2
import os
import csv
import time
import pickle

from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

video = cv2.VideoCapture(0)  # 0 is for webcam
if not video.isOpened():
    print("Error: Could not access the webcam")
    exit()
ret, frame = video.read()   # ret, frame are going to read web camera
if not ret:
    print("Error: Failed to capture image")
    exit()

face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  # OpenCV uses haar Cascade Classifier to detect the faces in the webcam

with open('Data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('Data/face_data,pkl', 'rb') as f:
    FACES = pickle.load(f)

print("Shape of FACES:", FACES.shape)
print("Length of LABELS:", len(LABELS))

# Fix mismatch if necessary
if len(FACES) != len(LABELS):
    # If LABELS has more samples than FACES, slice it to match
    LABELS = LABELS[:len(FACES)]  # or adjust this logic depending on the mismatch

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

img_background = cv2.imread("bg.jpg")
height, width, channels = img_background.shape
# print(f"Height: {height}, Width: {width}, Channels: {channels}")

COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()   # ret, frame are going to read web camera
    if not ret:
        print("Error: Failed to capture image")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)  # Using the scaling functions

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w, :]
        # Resize to match training dimensions
        resized_img = cv2.resize(crop_img, dsize=(25, 50))
        # Flatten and reshape
        flattened_img = resized_img.flatten().reshape(1, -1)
        # print("Prediction features:", flattened_img.shape[1])

        output = knn.predict(flattened_img)

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        exist = os.path.isfile("Attendance/Attendance_"+date+timestamp+".csv")
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 255), 2)

        attendance = [str(output[0]), str(timestamp)]

        height, width = img_background.shape[:2]
        frame_resized = cv2.resize(frame, (width, height))
        cv2.imshow("frame", frame_resized)

        k = cv2.waitKey(1)
        if k == ord('o'):
            time.sleep(5)

            if exist:
                with open("Attendance/Attendance_"+date+timestamp+".csv", 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(attendance)
                csvfile.close()
            else:
                with open("Attendance/Attendance_"+date+timestamp+".csv", 'w') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(COL_NAMES)
                    writer.writerow(attendance)
                csvfile.close()

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()

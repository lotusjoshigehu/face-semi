import os
import pickle
import numpy as np
import cv2
import face_recognition
import cvzone
import firebase_admin
from firebase_admin import credentials, db, storage
from datetime import datetime

# Initialize Firebase
cred = credentials.Certificate("serviceaccount.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendencerealtime-f66ae-default-rtdb.firebaseio.com/",
    'storageBucket': "faceattendencerealtime-f66ae.appspot.com"
})
bucket = storage.bucket()

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load background image
imgBackground = cv2.imread(r'C:\Users\joshi\Downloads\Screenshot 2024-07-11 143430.png')

# Load known images
folderModelPath = 'pictures'
modelPathList = os.listdir(folderModelPath)
imageModeList = [cv2.imread(os.path.join(folderModelPath, path), 1) for path in modelPathList]

# Load encoded faces
with open('encodedFile.p', 'rb') as file:
    encodeListunknownwithIds = pickle.load(file)
encodeListknown, studentIds = encodeListunknownwithIds

# Initialize variables
modeType = 0
counter = 0
id = -1
imgStudent = []
frame_skip = 5
frame_count = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from webcam. Exiting...")
        break

    resized_img = cv2.resize(img, (640, 470))
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurframe = face_recognition.face_encodings(imgS, faceCurFrame)
    imgBackground[162:162 + 470, 55:55 + 640] = resized_img

    if imageModeList:
        imgMode = imageModeList[modeType]
        imgMode = cv2.resize(imgMode, (472, 662))
        imgBackground[25:25 + imgMode.shape[0], 775:775 + imgMode.shape[1]] = imgMode

    if faceCurFrame:
        for encodeFace, Faceloc in zip(encodeCurframe, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListknown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListknown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                y1, x2, y2, x1 = Faceloc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                bbox = (55 + x1, 162 + y1, x2 - x1, y2 - y1)
                cvzone.cornerRect(imgBackground, bbox, rt=0)
                id = studentIds[matchIndex]
                print(f"Face detected. ID: {id}")

                if counter == 0:
                    counter = 1
                    modeType = 1

        if counter != 0:
            if counter == 1:
                studentInfo = db.reference(f'Students/{id}').get()
                print(f"Student info: {studentInfo}")
                blob = bucket.get_blob(f'pictures1/{id}.png')
                array = np.frombuffer(blob.download_as_string(), np.uint8)
                imgStudent = cv2.imdecode(array, cv2.IMREAD_COLOR)
                imgStudent = cv2.resize(imgStudent, (240, 280))
                datetimeObject = datetime.strptime(studentInfo['last_attendance_time'], "%Y-%m-%d %H:%M:%S")
                secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                if secondsElapsed > 300:  # Check if more than 30 minutes have passed
                    ref = db.reference(f'Students/{id}')
                    studentInfo['total attendance'] += 1
                    ref.child('total attendance').set(studentInfo['total attendance'])
                    ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    print("Attendance marked")
                else:
                    modeType = 3  # Set mode to already marked
                    counter = 0
                    imgBackground[25:25 + imgMode.shape[0], 775:775 + imgMode.shape[1]] = imgMode
                    print("Already marked")

            if modeType != 3:
                if 10 < counter < 20:
                    modeType = 2
                if counter <= 10:
                    cv2.putText(imgBackground, str(studentInfo['total attendance']), (830, 105),
                                cv2.FONT_ITALIC, 1, (255, 255, 255), 3)
                    cv2.putText(imgBackground, str(studentInfo['name']), (910, 455),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (100, 100, 100), 2)
                    cv2.putText(imgBackground, str(studentInfo['major']), (1010, 565),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(imgBackground, str(id), (1006, 504),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(imgBackground, str(studentInfo['standing']), (890, 640),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 2)
                    cv2.putText(imgBackground, str(studentInfo['year']), (1025, 639),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 2)
                    cv2.putText(imgBackground, str(studentInfo['starting year']), (1140, 637),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 2)
                    imgBackground[135:135 + 280, 895:895 + 240] = imgStudent
                counter += 1
                if counter >= 20:
                    counter = 0
                    modeType = 0
                    studentInfo = []
                    imgBackground[25:25 + imgMode.shape[0], 775:775 + imgMode.shape[1]] = imgMode
    else:
        modeType = 0
        counter = 0

    cv2.imshow('face Attendance', imgBackground)
    cv2.waitKey(1)
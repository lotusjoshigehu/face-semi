import pickle
import cv2
import face_recognition
import os

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
cred = credentials.Certificate("serviceaccount.json")
firebase_admin.initialize_app(cred,{'databaseURL': "https://faceattendencerealtime-f66ae-default-rtdb.firebaseio.com/",
                              'storageBucket':"faceattendencerealtime-f66ae.appspot.com"})

folderPath = 'pictures1'
pathList = os.listdir(folderPath)
print(pathList)
imageList = []
studentIds = []
for path in pathList:
    filePath = os.path.join(folderPath, path)
    image = cv2.imread(filePath)
    imgrgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imageList.append(imgrgb)
    studentId = os.path.splitext(path)[0]

    studentIds.append(studentId)

    fileName =f'{folderPath}/{path}'
    bucket=storage.bucket()
    blob=bucket.blob(fileName)
    blob.upload_from_filename(fileName)


print("Student IDs:", studentIds)

def findEncodings(imageList):
    encodeList = []
    for image in imageList:
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            encodeList.append(face_encodings[0])
    return encodeList

print("Encoding started...")
encodeListknown = findEncodings(imageList)
print("Encoding finished.")

encodeListunknownwithIds = [encodeListknown, studentIds]

file = open("encodedFile.p", 'wb')
pickle.dump(encodeListunknownwithIds, file)
file.close()
print("File saved as encodedFile.p")

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceaccount.json")
firebase_admin.initialize_app(cred, {'databaseURL': "https://faceattendencerealtime-f66ae-default-rtdb.firebaseio.com/"})
ref = db.reference("Students")

data = {
    "12346": {
        "name": "Karan Chauhan",
        "major": "Cs",
        "starting year": 2022,
        "total attendance": 0,
        "standing": "Good",
        "year": 3,
        "last_attendance_time": '2024-07-12 00:54:34',
        "photo_filename": "12346.png" # Adding the photo filename
    },
    "12345": {
        "name": "Vicky Kaushal",
        "major": "Acting",
        "starting year": 2022,
        "total attendance": 0,
        "standing": "bad",
        "year": 2,
        "last_attendance_time": '2024-07-12 00:54:34',
        "photo_filename": "12345.png"  # Adding the photo filename
    },
    "12344": {
        "name": "Kamal Joshi",
        "major": "Cs",
        "starting year": 2022,
        "total attendance": 0,
        "standing": "Good",
        "year": 3,
        "last_attendance_time": '2024-07-12 00:54:34',
        "photo_filename": "12344.png"  # Adding the photo filename
    }
}

# Upload the data to Firebase
for key, value in data.items():
    ref.child(key).set(value)

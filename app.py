from flask import Flask, render_template, Response, request, jsonify, url_for
import cv2
import face_recognition
import pickle
import firebase_admin
from firebase_admin import credentials, db
import numpy as np
from datetime import datetime, timedelta

# Initialize Flask
app = Flask(__name__)

# Firebase Initialization
cred = credentials.Certificate("serviceaccount.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendencerealtime-f66ae-default-rtdb.firebaseio.com/"
})

# Load face encodings
with open('encodedFile.p', 'rb') as file:
    encodeListunknownwithIds = pickle.load(file)
encodeListknown, studentIds = encodeListunknownwithIds

# Global Variables
cap = None
attendance_marked = []  # Initialize the attendance list


def gen_frames():
    """Generate frames for live video feed."""
    global cap
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Stream the video feed."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_attendance', methods=['POST'])
def start_attendance():
    """Start attendance marking."""
    global cap, attendance_marked
    if not cap or not cap.isOpened():
        return jsonify({"error": "Camera is not started!"}), 400

    success, frame = cap.read()
    if not success:
        return jsonify({"error": "Failed to capture frame!"}), 500

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    student_details = []
    unknown_faces_detected = False

    for encodeFace, faceLoc in zip(encodings, face_locations):
        matches = face_recognition.compare_faces(encodeListknown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListknown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            student_id = studentIds[matchIndex]
            student_info = db.reference(f'Students/{student_id}').get()

            # Check if the student's attendance has been marked in the last 30 minutes
            if 'last_attendance_time' in student_info:
                last_attendance_time = datetime.strptime(student_info['last_attendance_time'], "%Y-%m-%d %H:%M:%S")
                time_diff = datetime.now() - last_attendance_time

                if time_diff < timedelta(minutes=1):
                    # Attendance already marked in the last 30 minutes
                    continue

            # Update the last attendance time
            student_info['last_attendance_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Optionally update the attendance count
            student_info['total attendance'] = student_info.get('total attendance', 0) + 1
            db.reference(f'Students/{student_id}').set(student_info)

            # Add the student to the attendance list
            if 'photo_filename' in student_info:
                student_info['photo_url'] = url_for('static', filename=f'picture10/{student_info["photo_filename"]}')
            else:
                student_info['photo_url'] = url_for('static', filename='picture10/unknown.jpg')  # Placeholder image

            attendance_marked.append(student_info['name'])
            student_details.append(student_info)

        else:
            # If no match is found, set unknown face detected flag
            unknown_faces_detected = True

    # If an unknown face is detected, update the UI message
    return jsonify({
        "last_student": student_info if student_details else None,
        "count": len(attendance_marked),  # Send the count of students marked as present
        "unknown_face": "Unknown face detected" if unknown_faces_detected else None
    })


@app.route('/get_student_list', methods=['GET'])
def get_student_list():
    """Get the list of students marked present in the current session."""
    global attendance_marked
    return jsonify({"attendance": attendance_marked})



if __name__ == "__main__":
    app.run(debug=True)

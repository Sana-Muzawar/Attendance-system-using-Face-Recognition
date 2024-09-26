import cv2
import face_recognition
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

# Load face encodings from the file
with open('face_encodings.pkl', 'rb') as file:
    known_face_encodings, known_face_names = pickle.load(file)

# Initialize an empty list for attendance records
attendance_records = []

# Open webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    # Check if frame is captured successfully
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
    # rgb_frame = frame[:, :, ::-1]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    
    # if len(face_locations) > 0:
    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    else:
        face_encodings = []
    
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Check if the face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        # Calculate distances to find the best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        # Mark attendance
        if name != "Unknown":
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            
            # Check if the student is already marked present
            if name not in [record['Name'] for record in attendance_records]:
                attendance_records.append({'Name': name, 'Time': current_time})
                print(f"Attendance marked for: {name} at {current_time}")
        
        # Draw a rectangle around the face
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Attendance System', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Convert attendance records to a DataFrame and save to a CSV file
attendance_df = pd.DataFrame(attendance_records)
attendance_df.to_csv('attendance.csv', index=False)

# Release webcam and close windows
video_capture.release()
cv2.destroyAllWindows()

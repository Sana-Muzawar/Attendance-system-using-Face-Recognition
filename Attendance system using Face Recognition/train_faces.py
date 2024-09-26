import os
import face_recognition
import numpy as np
import pickle

# Path to dataset containing subfolders of images for each student
dataset_path = 'dataset'

# Initialize empty arrays to store encodings and names
known_face_encodings = []
known_face_names = []

# Loop through each person in the dataset
for student_name in os.listdir(dataset_path):
    student_folder = os.path.join(dataset_path, student_name)
    
    # Loop through each image file in the student's folder
    for image_name in os.listdir(student_folder):
        image_path = os.path.join(student_folder, image_name)
        
        # Load the image using face_recognition
        image = face_recognition.load_image_file(image_path)
        
        # Encode the face(s) in the image
        face_encodings = face_recognition.face_encodings(image)
        
        # Check if there's at least one face encoding found
        if face_encodings:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(student_name)

# Save encodings to a file for later use
with open('face_encodings.pkl', 'wb') as file:
    pickle.dump((known_face_encodings, known_face_names), file)

print("Training completed and encodings saved.")

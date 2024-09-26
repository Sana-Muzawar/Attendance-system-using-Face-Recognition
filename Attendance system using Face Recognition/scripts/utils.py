import os
import pickle
import pandas as pd
from datetime import datetime

def load_face_encodings(encoding_file='encodings/face_encodings.pkl'):
    """
    Load face encodings and names from a pickle file.

    Args:
        encoding_file (str): Path to the file containing face encodings.

    Returns:
        known_face_encodings (list): List of known face encodings.
        known_face_names (list): List of names corresponding to the encodings.
    """
    if os.path.exists(encoding_file):
        with open(encoding_file, 'rb') as file:
            known_face_encodings, known_face_names = pickle.load(file)
        return known_face_encodings, known_face_names
    else:
        print(f"Error: The encoding file {encoding_file} does not exist.")
        return [], []

def mark_attendance(name, attendance_file='attendance.csv'):
    """
    Mark attendance for a recognized student.

    Args:
        name (str): Name of the student to mark attendance for.
        attendance_file (str): Path to the attendance file (CSV).
    """
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    
    # Create a new DataFrame if the attendance file doesn't exist
    if not os.path.exists(attendance_file):
        df = pd.DataFrame(columns=['Name', 'Time'])
    else:
        df = pd.read_csv(attendance_file)

    # Check if the student is already marked present
    if name not in df['Name'].values:
        df = df.append({'Name': name, 'Time': current_time}, ignore_index=True)
        df.to_csv(attendance_file, index=False)
        print(f"Attendance marked for: {name} at {current_time}")
    else:
        print(f"{name} is already marked present.")

def create_directories():
    """
    Create necessary directories for the project if they do not exist.
    """
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
    if not os.path.exists('encodings'):
        os.makedirs('encodings')
    print("Directories are set up.")

def preprocess_image(image_path):
    """
    Load and preprocess an image for face encoding.

    Args:
        image_path (str): Path to the image file.

    Returns:
        numpy array: Preprocessed image suitable for face recognition.
    """
    try:
        # Load the image using face_recognition
        image = face_recognition.load_image_file(image_path) # type: ignore
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

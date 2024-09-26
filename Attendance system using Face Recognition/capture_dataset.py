import cv2
import os

def capture_images(student_name, num_images=20):
    """
    Capture images from the webcam and store them in a folder for the specified student.

    Args:
        student_name (str): The name of the student (used to create the subfolder).
        num_images (int): Number of images to capture for the dataset.
    """
    # Create a directory for the student if it doesn't exist
    dataset_path = f'dataset/{student_name}'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # Access the webcam (0 is usually the default camera)
    video_capture = cv2.VideoCapture(0)

    print(f"Capturing images for {student_name}. Press 'q' to quit.")

    count = 0
    while count < num_images:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Save the captured image to the dataset folder
        image_path = os.path.join(dataset_path, f'{student_name}_{count + 1}.jpg')
        cv2.imwrite(image_path, frame)
        count += 1
        print(f"Captured image {count}/{num_images}")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    video_capture.release()
    cv2.destroyAllWindows()
    print(f"Captured {count} images for {student_name} and saved them to {dataset_path}.")

if __name__ == "__main__":
    student_name = input("Enter the student's name: ")
    capture_images(student_name)

import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils

# Fixed path to the CSV file
csv_filename = r"C:\Users\Admin\Downloads\face_001\face_data\face_data.csv"

# Function to input name with a requirement that it cannot be empty
def input_name():
    while True:
        name = input("Enter a name for the person: ").strip()
        if name:
            return name
        print("Name cannot be blank. Please enter a valid name.")

# Prompt user for name
name = input_name()

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

input_count = 0
max_inputs = 300  # Number of inputs to collect for each face expression

# Check if CSV file exists, if not create it and write the header
try:
    if not os.path.exists(csv_filename):
        with open(csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            header = ['label'] + [f'x{i}' for i in range(468)] + [f'y{i}' for i in range(468)] + [f'z{i}' for i in range(468)]
            writer.writerow(header)
            
except Exception as e:
    print(f"Error: Could not open or create CSV file. {e}")
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    exit()

# Main loop to capture and process webcam feed
while cap.isOpened() and input_count < max_inputs:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image from webcam.")
        break

    frame = cv2.flip(frame, 1)

    # Convert color from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Detect face landmarks
    results = face_mesh.process(image)

    # Draw landmarks if faces are detected
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1))

            # Get landmarks data and save to CSV file
            landmarks = [name]
            for landmark in face_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            try:
                with open(csv_filename, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(landmarks)
            except Exception as e:
                print(f"Error: Could not write to CSV file. {e}")
                cap.release()
                cv2.destroyAllWindows()
                face_mesh.close()
                exit()
            
            input_count += 1
            print(f"save {input_count}/{max_inputs} model for face")

    # Display the frame
    cv2.imshow('Face Mesh Tracking', image)
    
    # Exit loop if Esc key is pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
face_mesh.close()

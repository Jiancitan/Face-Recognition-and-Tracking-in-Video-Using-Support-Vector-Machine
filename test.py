import cv2
import joblib
import mediapipe as mp

# Load trained SVM model and label encoder
model_filename = r"C:\Users\Admin\Downloads\face_001\svm_face_model.pkl"
label_encoder_filename = r"C:\Users\Admin\Downloads\face_001\label_encoder.pkl"

svm_model = joblib.load(model_filename)
label_encoder = joblib.load(label_encoder_filename)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Confidence threshold for unknown label
confidence_threshold = 0.85

# Main loop to capture and process webcam feed
while cap.isOpened():
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

    # Draw landmarks and bounding boxes if faces are detected
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Calculate bounding box
            h, w, _ = frame.shape
            x_min = min([int(landmark.x * w) for landmark in face_landmarks.landmark])
            x_max = max([int(landmark.x * w) for landmark in face_landmarks.landmark])
            y_min = min([int(landmark.y * h) for landmark in face_landmarks.landmark])
            y_max = max([int(landmark.y * h) for landmark in face_landmarks.landmark])
            
            # Draw bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Get landmarks data and predict using SVM model
            landmarks = [landmark for landmark in face_landmarks.landmark]
            flat_landmarks = [item for sublist in landmarks for item in [sublist.x, sublist.y, sublist.z]]
            prediction = svm_model.predict_proba([flat_landmarks])[0]
            
            # Get the label with highest confidence
            max_confidence_index = prediction.argmax()
            max_confidence = prediction[max_confidence_index]
            predicted_label = label_encoder.inverse_transform([max_confidence_index])[0]

            # Calculate predicted accuracy
            predicted_accuracy = max_confidence * 100

            # Check if confidence is above threshold
            if max_confidence >= confidence_threshold:
                # Display predicted label inside the bounding box
                cv2.putText(image, f'{predicted_label} ({predicted_accuracy:.2f}%)', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                # Display "unknown" label if confidence is below threshold
                cv2.putText(image, f'unknown ({predicted_accuracy:.2f}%)', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Face Recognition', image)

    # Exit loop if Esc key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
face_mesh.close()

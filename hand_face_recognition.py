import cv2
import numpy as np
import mediapipe as mp
import face_recognition
from tensorflow.keras.models import load_model
import os

# Initialize MediaPipe hands and drawing utilities
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the trained hand gesture model
model_path = 'C:\\Users\\asbi1\\PycharmProjects\\hand_face_recognition\\models\\mp_hand_gesture.h5'
model = load_model(model_path)

# Load gesture class names
gesture_names_path = 'C:\\Users\\asbi1\\PycharmProjects\\hand_face_recognition\\gesture.names'
with open(gesture_names_path, 'r') as f:
    gesture_class_names = f.read().split('\n')

# Load known faces and their encodings
known_face_encodings = []
known_face_names = []
known_faces_dir = 'C:\\Users\\asbi1\\PycharmProjects\\hand_face_recognition\\known_faces'

for filename in os.listdir(known_faces_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:  # Check if face encodings are found
            encoding = encodings[0]
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(filename)[0])

print("Known faces loaded:", known_face_names)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for face recognition
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    hand_class_name = ''

    # Post process the hand result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * frame.shape[1])
                lmy = int(lm.y * frame.shape[0])
                landmarks.append([lmx, lmy])

            # Drawing hand landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

        # Predict hand gesture
        if landmarks:
            prediction = model.predict([landmarks])
            classID = np.argmax(prediction)
            hand_class_name = gesture_class_names[classID]

    # Face detection and recognition
    face_locations = face_recognition.face_locations(framergb)
    face_encodings = face_recognition.face_encodings(framergb, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Find the best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # Display the name below the face
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the hand gesture name
    cv2.putText(frame, hand_class_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()

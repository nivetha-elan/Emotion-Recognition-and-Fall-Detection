import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained fall detection model
fall_detection_model = load_model('C:\\Users\\Nivi-HP\\Desktop\\Human+fall+Emotion\\trained models\\final_cnn_model.h5')

# Load pre-trained emotion recognition model
emotion_model = load_model('C:\\Users\\Nivi-HP\\Desktop\\Human+fall+Emotion\\trained models\\emotion_model.hdf5')

# Load labels for emotion recognition
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Open a video capture object
cap = cv2.VideoCapture(0)

# Set fall detection and emotion detection thresholds (adjust as needed)
fall_detection_threshold = 0.4
emotion_confidence_threshold = 0.5

# Initialize variables for fall detection history and person position
fall_history = []
person_position = None

while True:
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Reset fall flag for the current frame
    fall_detected = False

    for (x, y, w, h) in faces:
        # Update person position
        person_position = (x, y, w, h)

        # Extract the region of interest (ROI) containing the face
        face_roi = frame[y:y + h, x:x + w]

        # Preprocess face_roi for fall detection model
        face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi_resized = cv2.resize(face_roi_gray, (3, 21))
        face_roi_preprocessed = np.expand_dims(face_roi_resized, axis=-1)

        # Predict fall using the preprocessed data
        fall_prediction = fall_detection_model.predict(face_roi_preprocessed.reshape(1, 3, 21, 1))

        if fall_prediction > fall_detection_threshold:
            fall_detected = True
            cv2.putText(frame, "Fall Undetected!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Record fall detection in history
            fall_history.append(fall_detected)

    # Consider fall detection over recent frames
    if len(fall_history) >= 10 and sum(fall_history[-10:]) >= 6:
        # Check if the person is not present in the current frame
        person_present = any((x <= person_x <= x + w and y <= person_y <= y + h) for person_x, person_y, _, _ in faces)
        if not person_present:
            cv2.putText(frame, "Fall Confirmed!", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Update the person's position based on the first detected face
    # Update the person's position based on the first detected face
    if len(faces) > 0:
       x, y, w, h = faces[0]
       person_position = (x, y, w, h)


    # Recognize emotion
    face_roi_emotion = cv2.resize(frame[y:y + h, x:x + w], (64, 64))  # Crop face region for emotion recognition
    face_roi_emotion = cv2.cvtColor(face_roi_emotion, cv2.COLOR_BGR2GRAY)
    face_roi_emotion = np.expand_dims(face_roi_emotion, axis=-1)
    face_roi_emotion = face_roi_emotion / 255.0  # Normalize to [0, 1]

    # Predict using the preprocessed data
    emotion_predictions = emotion_model.predict(np.expand_dims(face_roi_emotion, axis=0))

    # Get the predicted emotion label and confidence
    emotion_label = emotion_labels[np.argmax(emotion_predictions)]
    emotion_confidence = np.max(emotion_predictions)

    # Display emotion label on the frame if confidence is above the threshold
    if emotion_confidence > emotion_confidence_threshold:
        cv2.putText(frame, f"Emotion: {emotion_label}", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255 , 0 , 0), 2)

    # Draw rectangle around the face
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Human Recognition, Fall Detection, and Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()


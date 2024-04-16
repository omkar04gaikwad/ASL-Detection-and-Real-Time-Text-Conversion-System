import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from collections import deque
import statistics

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # Drawing helpers

# Load the JSON file that contains the model's architecture
with open('pose_estimation.json', 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)

# Load your trained model
try:
    model.load_weights('pose_estimation.weights.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def extract_features(hand_landmarks):
    """Extract hand pose features from the landmarks."""
    try:
        return np.array([(landmark.x, landmark.y) for landmark in hand_landmarks.landmark]).flatten()
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.zeros(42)

def classify_sign(features):
    """Predict the hand sign from the extracted features."""
    try:
        features = np.expand_dims(features, axis=0)  # Model expects a batch of examples
        predictions = model.predict(features)
        predicted_class_index = np.argmax(predictions)
        predicted_class_probability = predictions[0][predicted_class_index]
        predicted_letter = chr(predicted_class_index + ord('a'))
        return predicted_letter
    except Exception as e:
        print(f"Error during classification: {e}")
        return -1

def main():
    cap = cv2.VideoCapture(0)  # Capture video from camera
    gesture_buffer = deque(maxlen=15)  # Buffer to store recent predictions for smoothing
    current_sign = None
    sign_name = ""  # Initialize the sign name to be displayed

    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB and process with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                
                features = extract_features(hand_landmarks)
                sign_id = classify_sign(features)
                gesture_buffer.append(sign_id)

                # Process gesture identification
                if len(gesture_buffer) == gesture_buffer.maxlen:
                    # Find the most common gesture in the buffer
                    try:
                        most_common = statistics.mode(gesture_buffer)
                        if most_common != current_sign:
                            current_sign = most_common
                            sign_name = f'Sign ID: {current_sign}'  # Update the sign name
                    except statistics.StatisticsError:
                        continue  # Continue if no unique mode is found
        else:
            gesture_buffer.clear()
            current_sign = None  # Clear the current sign if no hand is detected

        # Display the sign name continuously
        if current_sign:
            cv2.putText(frame, sign_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Hand Sign Recognition', frame)
        if cv2.waitKey(5) & 0xFF == 27:  # Exit on ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

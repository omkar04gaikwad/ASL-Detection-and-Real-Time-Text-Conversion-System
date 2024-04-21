import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import time

# Initialize the state variables
if 'sentence' not in st.session_state:
    st.session_state['sentence'] = ""
if 'last_sign' not in st.session_state:
    st.session_state['last_sign'] = None
if 'sign_start_time' not in st.session_state:
    st.session_state['sign_start_time'] = None
if 'displayed_sign' not in st.session_state:
    st.session_state['displayed_sign'] = None

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load the model
model_path = 'pose_estimation.weights.h5'
model_json = 'pose_estimation.json'

with open(model_json, 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights(model_path)

def extract_features(hand_landmarks):
    try:
        return np.array([(landmark.x, landmark.y) for landmark in hand_landmarks.landmark]).flatten()
    except Exception as e:
        return np.zeros(42)

def classify_sign(features):
    features = np.expand_dims(features, axis=0)
    predictions = model.predict(features)
    predicted_class_index = np.argmax(predictions)
    return chr(predicted_class_index + ord('a'))

def main():
    st.title("Hand Sign Recognition App")

    # Layout adjustments
    col1, col2 = st.columns(2)
    with col1:
        FRAME_WINDOW = st.empty()  # Placeholder for the video feed
    with col2:
        predicted_letter = st.empty()  # Placeholder for predicted letter
        sentence_display = st.empty()  # Placeholder for sentence display
        if st.button('Clear Word'):  # Button to clear the word
            st.session_state['sentence'] = ""
        if st.button('Backspace'):  # Button to remove the last word
            if st.session_state['sentence']:
                st.session_state['sentence'] = st.session_state['sentence'][:-1]

    # Set the initial display so they don't disappear on clear
    predicted_letter.markdown(f'<h2 style="font-size:24px;">Predicted Letter:</h2>', unsafe_allow_html=True)
    sentence_display.markdown(f'<h2 style="font-size:24px;">Word:</h2>', unsafe_allow_html=True)

    cap = cv2.VideoCapture(0)  # Capture video from the camera

    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        if ret:
            results = mp_hands.process(frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                    features = extract_features(hand_landmarks)
                    current_sign = classify_sign(features)
                    if current_sign == st.session_state['last_sign'] and st.session_state['sign_start_time'] is not None:
                        if (time.time() - st.session_state['sign_start_time'] > 2) and (st.session_state['displayed_sign'] != current_sign):
                            st.session_state['sentence'] += current_sign
                            st.session_state['displayed_sign'] = current_sign
                    else:
                        st.session_state['last_sign'] = current_sign
                        st.session_state['sign_start_time'] = time.time()
                        st.session_state['displayed_sign'] = None

                    # Update the placeholders with current sign and sentence
                    predicted_letter.markdown(f'<h2 style="font-size:24px;">Predicted Letter: {current_sign}</h2>', unsafe_allow_html=True)
                    sentence_display.markdown(f'<h2 style="font-size:24px;">Word: {st.session_state["sentence"]}</h2>', unsafe_allow_html=True)

            FRAME_WINDOW.image(frame)
        else:
            st.write("Failed to read from camera.")
            break

    cap.release()

if __name__ == "__main__":
    main()

import cv2
import os
from sklearn.utils import shuffle
import numpy as np
import mediapipe as mp
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical

def load_data(directory):
    images = []
    labels = []
    poses = []
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        for filename in os.listdir(label_path):
            image_path = os.path.join(label_path, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (100, 100))
            images.append(image)
            pose_features = extract_pose_features(image)
            if pose_features is not None:  # Check if pose features are extracted successfully
                poses.append(pose_features)
                labels.append(ord(label) - ord('A'))
            else:
                print(f"Pose estimation failed for image: {image_path}")
    return np.array(images), np.array(poses), np.array(labels)



# Function to extract pose features using MediaPipe
def extract_pose_features(image):
    mp_hands = mp.solutions.hands.Hands(static_image_mode=False)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(image_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        x_coords = [landmark.x for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y for landmark in hand_landmarks.landmark]
        pose_features = np.array(x_coords + y_coords)
        return pose_features
    else:
        return None

dataset_directory = 'dataSet/trainingData'

images, poses, labels = load_data(dataset_directory)

# Shuffle the data
images, poses, labels = shuffle(images, poses, labels, random_state=42)

# Data Preprocessing (Example: Normalize pixel values)
images = images.astype('float32') / 255.0

#convert labels to one-hot encoding
labels_one_hot = to_categorical(labels)
print("Images Shape:", images.shape)
print("Poses Shape:", poses.shape)
print("Labels Shape:", labels.shape)
# print("hello")
try:
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(36, activation='softmax')
    ])
    print("modeled")
except Exception as e:
    print("An error occurred during training:", e)
try:
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    print("compiled")
except Exception as e:
    print("An error occurred during training:", e)


# Print model summary
print(model.summary())

try:
    model_history = model.fit([images, poses], labels_one_hot, epochs=10, validation_split=0.2)
except Exception as e:
    print("An error occurred during training:", e)

# Load and preprocess test data
test_dataset_directory = 'dataSet/testingData'
test_images, test_poses, test_labels = load_data(test_dataset_directory)
test_images, test_poses, test_labels = shuffle(test_images, test_poses, test_labels, random_state=42)
test_images = test_images.astype('float32') / 255.0
print("Images Shape:", test_images.shape)
print("Poses Shape:", test_poses.shape)
print("Labels Shape:", test_labels.shape)
# Convert test labels to one-hot encoding
test_labels_one_hot = to_categorical(test_labels)
# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate([test_images, test_poses], test_labels_one_hot)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)



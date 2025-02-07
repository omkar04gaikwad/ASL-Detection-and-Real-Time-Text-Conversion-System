# ASL Prediction Using Pose Estimation

## Overview
This project focuses on **real-time American Sign Language (ASL) prediction** using **Pose Estimation** techniques. It employs **Convolutional Neural Networks (CNNs)** combined with **MediaPipe's 2D pose estimation** to accurately classify hand gestures into ASL alphabets. The primary objective is to enhance communication accessibility for individuals who are **deaf or hard of hearing**, allowing seamless interaction with non-ASL users.

## Tech Stack
- **Machine Learning Framework**: TensorFlow, Keras
- **Pose Estimation**: MediaPipe
- **Computer Vision**: OpenCV
- **Frontend**: Streamlit (for real-time gesture visualization)
- **Programming Language**: Python
- **Dataset**: Custom dataset with 8000+ images per alphabet of ASL alphabets
- **Deployment**: Local execution with potential for cloud deployment

## System Architecture
```
+---------------------+       +----------------------+       +----------------+
|  Pose Estimation  | ----> |   Feature Extraction | ----> |   CNN Model    |
+---------------------+       +----------------------+       +----------------+
          |                             |                              |
+--------------------+       +---------------------+       +----------------+
|   Streamlit UI   |       | OpenCV Processing |       |   Model Inference |
+--------------------+       +---------------------+       +----------------+
```
### **Component Breakdown**
1. **Pose Estimation (MediaPipe)**: Detects hand keypoints and extracts features.
2. **Feature Extraction**: Converts pose keypoints into a structured feature vector.
3. **CNN Model (Keras)**: Trained to classify gestures based on extracted pose features.
4. **Streamlit UI**: Real-time user interface for gesture recognition and translation.
5. **OpenCV Processing**: Handles video capture and preprocessing.

---
## **Dataset and Preprocessing**
- **Custom ASL Dataset**:
  - **Training Data**: 5800 images
  - **Validation Data**: 1100 images
  - **Testing Data**: 1100 images
  - **Image Size**: 32x32 pixels
- **Feature Extraction**:
  - Uses **MediaPipe Hand Tracking** to extract **(x, y) coordinates** of hand landmarks.
  - Normalized and structured into a fixed-length feature vector.
  - Stored in a **Parquet file** for faster processing.

---
## **Model Training and Performance**
- **Model Architecture**:
  - Input Layer: 42-feature vector (21 keypoints, x and y)
  - Hidden Layers: Fully connected layers with ReLU activation and Dropout
  - Output Layer: Softmax activation for **26-class classification (A-Z)**
- **Training Parameters**:
  - **Optimizer**: Adam
  - **Loss Function**: Categorical Crossentropy
  - **Epochs**: 10
  - **Batch Size**: 32
- **Performance Metrics**:
  - Training Accuracy: **98.77%**
  - Validation Accuracy: **97.41%**
  - Test Accuracy: **97.66%**

---
## **Software Implementation**
### **1️⃣ Streamlit Web Interface**
- Displays real-time video feed from a webcam.
- Detects and classifies ASL gestures dynamically.
- Provides a **text output of recognized words**.

### **2️⃣ OpenCV Processing**
- Captures live webcam feed.
- Preprocesses frames (flipping, color conversion, and resizing).
- Draws detected hand landmarks on frames.

### **3️⃣ Model Inference**
- Extracts pose keypoints from each video frame.
- Runs the feature vector through the **trained CNN model**.
- Outputs the recognized ASL letter.

---
## **Endpoints Documentation**
- **POST `/predict`** - Accepts an image/frame and returns the predicted ASL letter.
- **GET `/visualize`** - Streams real-time video feed with ASL classification.
- **POST `/train`** - Retrains the model with new ASL data.

---
## **Setup & Run the Project**
### **1️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```
### **2️⃣ Initialize Dataset & Preprocessing**
```bash
python preprocess_data.py  # Extracts features and saves as Parquet
```
### **3️⃣ Train the Model**
```bash
python train_model.py  # Trains and saves CNN model
```
### **4️⃣ Run Streamlit Web App**
```bash
streamlit run app.py
```
### **5️⃣ Run Tests**
```bash
pytest tests/
```

---
## **Challenges Faced**
- **Similar Gesture Confusion**: The model sometimes misclassifies visually similar ASL signs (e.g., 'R' vs. 'U').
- **Dataset Imbalance**: Majority right-hand data led to lower accuracy for left-handed gestures.
- **Real-Time Processing Lag**: Optimized MediaPipe and CNN inference to reduce latency.

---
## **Future Enhancements**
- **Expand Dataset**: Include more variations of ASL gestures for robustness.
- **Deploy on Cloud**: Host the model on AWS/GCP for online accessibility.
- **Mobile Integration**: Develop a mobile app version for real-time ASL translation.
- **Multi-User Support**: Allow multiple users to interact simultaneously.

---

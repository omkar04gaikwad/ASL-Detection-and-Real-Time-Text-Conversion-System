# 🤟 ASL Detection and Real-Time Text Conversion System

## 🔍 Overview

This project implements a **real-time American Sign Language (ASL) recognition system** using **Pose Estimation** and **Convolutional Neural Networks (CNNs)**. Unlike traditional image classification that processes static frames, this system leverages **MediaPipe's hand landmark detection** to extract dynamic gesture features and classify them into ASL alphabets.

A **Streamlit web application** serves as the user interface, enabling real-time translation of ASL gestures into text, providing a communication bridge between individuals who use ASL and those who may not be familiar with it.

## 🎯 Problem Statement

Communication barriers between deaf/hard-of-hearing individuals and non-ASL users create significant challenges in:
- **Educational settings** where interpreters may not be available
- **Healthcare environments** requiring clear communication
- **Workplace interactions** limiting career opportunities
- **Social situations** creating isolation and exclusion

This system addresses these challenges by providing an automated, real-time ASL-to-text translation solution that bridges communication gaps and promotes inclusivity.

## 🔑 Key Features

### **Advanced Pose Estimation**
- **MediaPipe Integration**: Extracts 2D hand landmarks (21 points per hand) to capture fine-grained gesture details
- **Dynamic Feature Extraction**: Processes real-time hand movements, not just static images
- **Robust Detection**: Works across different lighting conditions and hand positions

### **Custom Dataset & Training**
- **Massive Dataset**: Generated over **208,000 images** across ASL alphabets (32×32 resolution)
- **Comprehensive Coverage**: All 26 ASL alphabet letters (A-Z) with multiple variations
- **Efficient Storage**: Features stored in `.parquet` format for optimal preprocessing speed

### **Deep Learning Architecture**
- **Input Layer**: 42-dimensional landmark vectors (x, y coordinates)
- **Hidden Layers**: Dense layers (128 → 64) with ReLU activation + dropout (0.2)
- **Output Layer**: Softmax classification with 26 classes (A–Z)
- **Optimization**: Adam optimizer with categorical crossentropy loss

### **Real-Time Performance**
- **High Accuracy**: Achieved **97.6% test accuracy** with comprehensive validation
- **Live Processing**: Real-time detection on webcam feed with minimal latency
- **Stability Features**: 2-second confirmation delay to ensure gesture accuracy
- **Dynamic Text**: Live construction of words and sentences with editing capabilities

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Webcam        │───▶│   MediaPipe     │───▶│   Feature       │
│   Input         │    │   Hand          │    │   Extraction    │
│                 │    │   Landmarks     │    │   (42D Vector)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │◀───│   Text         │◀───│   CNN Model     │
│   Web Interface │    │   Conversion   │    │   Classification│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Tech Stack

### **Core Technologies**
- **🤖 Machine Learning**: TensorFlow, Keras
- **👁️ Computer Vision**: MediaPipe, OpenCV
- **🌐 Web Interface**: Streamlit
- **📊 Data Processing**: Pandas, NumPy
- **💾 Data Storage**: Parquet files for efficient preprocessing

### **Key Libraries**
```python
tensorflow==2.10.0
mediapipe==0.10.0
opencv-python==4.8.0
streamlit==1.28.0
pandas==1.5.0
numpy==1.24.0
```

## 📊 Dataset & Performance

### **Dataset Statistics**
| Component | Count | Details |
|----------|-------|---------|
| **Total Images** | 208,000+ | Custom ASL alphabet dataset |
| **Training Data** | 5,800 | Balanced across all 26 letters |
| **Validation Data** | 1,100 | Independent validation set |
| **Testing Data** | 1,100 | Final performance evaluation |
| **Image Resolution** | 32×32 | Optimized for speed and accuracy |

### **Model Performance**
| Metric | Score | Description |
|--------|-------|-------------|
| **Training Accuracy** | 98.77% | Model learning capability |
| **Validation Accuracy** | 97.41% | Generalization performance |
| **Test Accuracy** | 97.66% | Real-world performance |
| **Processing Speed** | Real-time | Live webcam feed processing |

## 🧠 Model Architecture

### **CNN Architecture Details**
```python
# Model Architecture
Input Layer:     42 features (21 keypoints × 2 coordinates)
Hidden Layer 1:  128 neurons + ReLU + Dropout(0.2)
Hidden Layer 2:  64 neurons + ReLU + Dropout(0.2)
Output Layer:    26 neurons + Softmax (A-Z classification)
```

### **Training Configuration**
- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 10 (with early stopping)
- **Validation Split**: 80/20

### **Feature Extraction Pipeline**
1. **Hand Detection**: MediaPipe hand tracking
2. **Keypoint Extraction**: 21 hand landmarks (x, y coordinates)
3. **Normalization**: Coordinate normalization and scaling
4. **Feature Vector**: 42-dimensional input vector
5. **Model Inference**: Real-time classification

## 📁 Project Structure

```
ASL-Detection-and-Real-Time-Text-Conversion-System/
├── 🤖 ASL_python/
│   ├── Dataset/                    # Training data (208,000+ images)
│   ├── models/                     # Trained model files
│   ├── preprocessing/              # Data preprocessing scripts
│   └── utils/                     # Utility functions
├── 🖥️ ASL_C++/
│   ├── Dataset/                   # C++ implementation data
│   └── src/                       # C++ source code
├── 🎯 ASL_pose/
│   ├── pose_estimation.py         # MediaPipe integration
│   ├── feature_extraction.py      # Keypoint processing
│   └── model_training.py          # CNN training pipeline
├── 📊 Models/
│   ├── asl_cnn_model.h5          # Trained CNN model
│   ├── model_metrics.json        # Performance metrics
│   └── training_history.pkl      # Training logs
├── 🌐 Streamlit App/
│   ├── app.py                     # Main web application
│   ├── real_time_detection.py    # Live ASL recognition
│   └── visualization.py          # Results display
├── 📋 Documentation/
│   ├── README.md                  # This file
│   ├── API_documentation.md       # API endpoints
│   └── deployment_guide.md        # Deployment instructions
└── 🖼️ signs.jpg                   # ASL alphabet reference
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **Webcam** (built-in or external)
- **4GB+ RAM**
- **Modern web browser**

### Installation

1. **Clone the Repository**
```bash
git clone https://github.com/omkar04gaikwad/ASL-Detection-and-Real-Time-Text-Conversion-System.git
cd ASL-Detection-and-Real-Time-Text-Conversion-System
```

2. **Install Dependencies**
```bash
pip install tensorflow==2.10.0
pip install mediapipe==0.10.0
pip install opencv-python==4.8.0
pip install streamlit==1.28.0
pip install pandas numpy matplotlib
```

3. **Run the Application**
```bash
# Start the Streamlit web app
streamlit run app.py

# Or run the real-time detection directly
python real_time_detection.py
```

4. **Access the Interface**
- Open your browser to `http://localhost:8501`
- Allow webcam permissions
- Start signing ASL letters!

## 🎮 Usage Guide

### Real-Time ASL Recognition
1. **Launch Application**: Start the Streamlit web interface
2. **Camera Setup**: Position your hand clearly in the webcam view
3. **Sign Letters**: Perform ASL alphabet gestures
4. **View Results**: See real-time text conversion on screen
5. **Build Words**: Construct words and sentences with detected letters

### Supported ASL Letters
The system recognizes all 26 ASL alphabet letters:
**A B C D E F G H I J K L M N O P Q R S T U V W X Y Z**

### Best Practices
- **Lighting**: Ensure good lighting for clear hand visibility
- **Background**: Use plain backgrounds for better detection
- **Hand Position**: Keep hands within the camera frame
- **Gesture Clarity**: Make clear, distinct ASL gestures
- **Stability**: Hold gestures for 2 seconds for confirmation

## 📈 Results & Evaluation

### **Performance Metrics**
- **Training Accuracy**: 98.77%
- **Validation Accuracy**: 97.41%
- **Test Accuracy**: 97.66%
- **Processing Latency**: Real-time detection on live webcam feed

### **Validation Methods**
- **Confusion Matrix**: Comprehensive classification analysis
- **Cross-Validation**: Robust performance evaluation
- **Real-Time Testing**: Live webcam feed validation
- **Edge Case Analysis**: Challenging gesture recognition

### **Current Limitations**
- **Similar Gesture Confusion**: Occasional misclassification of visually similar signs (e.g., 'R' vs 'U')
- **Hand Dominance Bias**: Reduced accuracy for left-handed users due to right-hand dominant training data
- **Lighting Sensitivity**: Performance varies with lighting conditions
- **Gesture Complexity**: Limited to single-hand alphabet recognition

## 🔧 API Endpoints

### REST API Documentation
```python
# POST /predict
# Accepts image/frame and returns predicted ASL letter
{
    "image": "base64_encoded_image",
    "confidence_threshold": 0.8
}

# GET /visualize
# Streams real-time video feed with ASL classification
# Returns: Live video stream with overlay

# POST /train
# Retrains the model with new ASL data
{
    "training_data": "path_to_new_data",
    "epochs": 10,
    "batch_size": 32
}
```

## 🎯 Use Cases

### Educational Applications
- **ASL Learning**: Interactive tool for learning sign language
- **Classroom Integration**: Real-time translation in educational settings
- **Language Training**: Practice and assessment platform
- **Accessibility Tools**: Supporting inclusive education

### Healthcare & Professional
- **Medical Communication**: Doctor-patient ASL translation
- **Emergency Services**: Quick communication in urgent situations
- **Workplace Integration**: Office communication support
- **Customer Service**: Inclusive service interactions

### Personal & Social
- **Family Communication**: Bridging communication gaps at home
- **Social Media**: ASL content creation and sharing
- **Gaming**: Accessible gaming interfaces
- **Entertainment**: ASL-based interactive applications

## 🚀 Future Work

### **Short-Term Enhancements**
- **Dataset Expansion**: Include dynamic signs and left-hand gestures
- **Image Augmentation**: Better generalization across diverse conditions
- **Mobile App**: iOS and Android applications for broader accessibility
- **UI/UX Improvements**: Mobile-friendly interface design

### **Long-Term Vision**
- **Full Word Recognition**: Extend from alphabets to complete words/phrases
- **3D Pose Estimation**: Higher accuracy for complex gestures
- **Multi-Language Support**: International sign language variations
- **Cloud Deployment**: Scalable online accessibility

### **Technical Roadmap**
- **Edge Computing**: On-device processing for privacy
- **Federated Learning**: Distributed model training
- **Real-Time Translation**: Live conversation translation
- **Accessibility Standards**: WCAG compliance and certification

## 🌍 Impact

This project demonstrates how **AI + Computer Vision** can bridge communication gaps for the deaf and hard-of-hearing community, providing accessible, real-time translation of sign language. The system has the potential to:

- **Enhance Accessibility**: Make communication more inclusive
- **Improve Education**: Support ASL learning and teaching
- **Enable Independence**: Reduce reliance on interpreters
- **Promote Inclusion**: Break down social barriers

## 👨‍💻 Contributors

- **Omkar Gaikwad** ([@omkar04gaikwad](https://github.com/omkar04gaikwad)) - Lead Developer
- **Parthesh** ([@partheshp](https://github.com/partheshp)) - Collaborator

## 📚 References

### Academic References
- **MediaPipe**: [Google Research, 2019](https://mediapipe.dev/)
- **Hand Pose Estimation**: [Zimmermann & Brox, 2017](https://arxiv.org/abs/1704.07809)
- **ASL Recognition**: [Koller et al., 2016](https://arxiv.org/abs/1602.07325)
- **Real-Time Computer Vision**: [Bradski & Kaehler, 2008](https://opencv.org/)

### Dataset Sources
- **Custom ASL Dataset**: 208,000+ hand-labeled images
- **ISL Dataset**: International Sign Language references
- **Academic Datasets**: Research-grade ASL gesture collections

## 🤝 Contributing

We welcome contributions to improve this project! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### How to Contribute
1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Areas for Contribution
- **Model Improvements**: Better accuracy and performance
- **UI/UX Enhancement**: Improved user interface design
- **Documentation**: Better guides and tutorials
- **Testing**: Comprehensive test coverage
- **Accessibility**: Enhanced accessibility features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Important Notes

### Privacy & Ethics
- **Data Privacy**: No personal data is stored or transmitted
- **Local Processing**: All analysis happens on your device
- **Consent**: Users control their data and participation
- **Accessibility**: Designed with inclusivity as the primary goal

### Limitations
- **Current Scope**: Limited to ASL alphabet recognition
- **Accuracy**: May vary based on lighting and hand positioning
- **Language Support**: Currently supports American Sign Language only
- **Hardware Requirements**: Requires webcam and adequate lighting

---

⭐ **Star this repository** if you find it helpful for accessibility and communication!

🔔 **Watch for updates** to stay informed about new features and improvements.

📧 **Contact**: For questions, collaborations, or accessibility feedback, please reach out through GitHub.

🌍 **Mission**: Breaking down communication barriers, one gesture at a time.

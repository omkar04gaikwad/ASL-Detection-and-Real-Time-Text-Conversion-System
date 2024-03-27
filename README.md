# ASL-Detection-and-Real-Time-Text-Conversion-System

## Proposal: 

This project focuses on creating an advanced American Sign Language (ASL) detection and real-time text conversion system using computer vision techniques. Our main goal is to demonstrate a nuanced understanding of computer vision principles, specifically emphasizing the nontrivial application of Convolutional Neural Networks (CNNs) and Pose Estimation. 

In pursuit of efficient ASL gesture recognition, we will leverage CNNs for feature extraction and classification, implementing preprocessing techniques like Gaussian Blur filtering and binary thresholding. The challenge lies in optimizing CNNs to ensure accurate and rapid gesture detection, moving beyond standard implementations. 

The alternative approach involves exploring ASL detection through Pose Estimation, adding depth to our understanding of gesture recognition methodologies. By achieving real-time operation, this project aims to showcase a comprehensive yet streamlined ASL detection system, combining the strengths of CNNs and Pose Estimation for practical applications. 

  

## Data: 

The project will require a dataset of American Sign Language (ASL) gestures captured in various environmental conditions both to train and test the enhanced model. To this end, we will generate a custom dataset using the OpenCV library, capturing approximately 800 images per symbol for training and 200 per symbol for testing. This dataset will include gestures presented against a blue-bounded square region of interest (ROI) to standardize the background, facilitating more accurate gesture recognition.  


## Run C++
1. Open Ubuntu
2. go to the address:
    cd /mnt/c/Users/omkar/OneDrive/Desktop/EECE5639/ASL_project/ASL-Detection-and-Real-Time-Text-Conversion-System/ASL_C++
3. run the program
    g++ -std=c++17 trainingdata.cpp
    ./a.out
    g++ -o output_file.out trainingdata.cpp -std=c++17 `pkg-config --cflags --libs opencv4`
# 🤟 Indian Sign Language Recognition Using Distributed Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.x-brightgreen.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange.svg)](https://www.tensorflow.org/)

Welcome to the **Indian Sign Language Recognition Using Distributed Machine Learning** project! This repository showcases an innovative project aimed at recognizing Indian Sign Language (ISL) using deep learning techniques distributed across multiple processors for faster and more efficient training.

## 🚀 Introduction

Communication barriers for the deaf and hard-of-hearing communities have always been a challenge. Indian Sign Language (ISL) is one such language that allows individuals to communicate using gestures. However, not everyone understands ISL, which creates a communication gap.

This project aims to bridge that gap by creating a **Sign Language Recognition (SLR) system** using **Distributed Machine Learning** to accelerate the training of deep learning models. The system leverages **Convolutional Neural Networks (CNNs)** and distributed computing with **TPUs** and **GPUs** to achieve high accuracy in recognizing hand gestures and translating them into English text.

## ❓ Problem Statement

The main objective of this project is to create a robust and efficient system for recognizing Indian Sign Language (ISL) using machine learning. The specific challenges addressed include:
- **Recognition Accuracy**: Ensuring high accuracy in detecting and translating ISL gestures.
- **Training Speed**: Using distributed machine learning techniques to accelerate model training with large datasets.
- **Scalability**: Making the system scalable for real-world applications, enabling faster deployment of models in production environments.

## 🔬 Methodology

The methodology followed in this project involves the following key steps:

1. **Data Collection and Preprocessing**: The dataset used contains 1000 images per ISL letter. Data preprocessing involves skin detection, edge detection, and segmentation to isolate hand gestures from the background.
2. **Model Creation**: A **Convolutional Neural Network (CNN)** was created to classify the hand gestures. Multiple convolutional layers were used, followed by dropout layers to prevent overfitting. The model is trained using categorical cross-entropy and optimized with the Adam optimizer.
3. **Distributed Training**: The model is trained using distributed computing techniques:
   - **TPUs**: Tensor Processing Units were used to parallelize the model training, significantly reducing training time.
   - **GPUs**: Graphical Processing Units were also used to compare the performance with TPU-based distributed learning.

![Methodology Flowchart](https://via.placeholder.com/800x300.png?text=Methodology+Flowchart)

## 📊 Results

The system achieved an accuracy of over **98%** across all distributed models, with significant speedups in training times when using TPUs and GPUs. The comparison of different training approaches showed that:
- **CPU-based Training**: Baseline accuracy, slowest training.
- **GPU-based Training**: Achieved a speedup of **4.63x** over the CPU model.
- **TPU-based Training**: Achieved a speedup of **11.32x** over the CPU model, making it the most efficient for distributed learning.

## ✨ Features

- **High Accuracy Recognition**: Recognizes ISL gestures with over 98% accuracy.
- **Distributed Learning**: Utilizes TPUs and GPUs for faster model training.
- **Real-Time Translation**: Translates ISL gestures into English text in real-time.
- **Scalable Solution**: Designed for scalability to handle larger datasets and more complex models in the future.

## 🛠️ Technology Stack

- **Python**: Programming language used for the entire project.
- **TensorFlow**: Deep learning framework used to build and train the CNN models.
- **OpenCV**: Library used for image processing tasks such as skin detection and edge detection.
- **Distributed Computing**: Utilized TPUs and GPUs for parallelized model training.

## 🛠️ Installation

To set up the project locally:

1. **Clone the repository**:
   ```bash
   [git clone https://github.com/your-username/indian-sign-language-recognition.git
   cd indian-sign-language-recognition](https://github.com/shri11081999/Indian-Sign-Language-Recognition-Using-Distributed-Machine-Learning.git)

2. Set up TensorFlow:
Ensure you have the correct version of TensorFlow installed that supports TPU and GPU acceleration.

## 🖥️ Usage

1. Run the training script.

2. Select the hardware accelerator:

Use TPU or GPU for distributed training.

3. Monitor Training: Visualize the training performance using TensorBoard or similar tools.

## 📸 Screenshots

![image](https://github.com/user-attachments/assets/0176122c-edf3-48db-a80e-9a55565b8e4a)

![image](https://github.com/user-attachments/assets/0826a537-3e44-43c0-b770-20c9ec6a61ef)

## 👥 Contributors
Shriniket Dixit (GitHub)
Pratham Gupta
Jinay Bafna

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

# cnn_classification
cnn_classification

## Rationale:
Alzheimer’s disease is a progressive brain disorder that is difficult to detect early, and current diagnosis methods can be slow and costly. This project aims to develop an AI model that can classify MRI scans as healthy or Alzheimer-affected, helping doctors identify the disease faster and cost-efficiently. Early detection can improve treatment outcomes, support patients and families, and make diagnostic tools more accessible everywhere.

## Research Question
Can a convolutional neural network (CNN) trained on public MRI Datasets accurately distinguish between MRI scans of non-Alzheimer's adults and those diagnosed with Alzheimer’s disease (binary classification)?

## Requirements:
The system must be able to take MRI (Magnetic Resonance Imaging) brain images as input and classify them as either belonging to a non-Alzheimer’s individual or a patient with Alzheimer’s disease. To ensure ethical use, the system will be trained only on publicly available, fully de-identified MRI datasets. The classification will be performed using a convolutional neural network (CNN), a deep-learning model specifically designed for image analysis. The system must achieve measurable performance, with a target accuracy of at least 80%, evaluated using metrics such as accuracy and precision.

The project must also include clear visual outputs to help interpret the results. These outputs will include training and validation accuracy graphs, a confusion matrix, and example MRI predictions. The model will be developed entirely in Python using Google Colab and free, publicly available machine learning libraries so that the project is easy to access and reproduce. A public Alzheimer’s MRI dataset (such as one from Kaggle) will be used, and no personal or patient-identifying information will be included.

All MRI images must be automatically loaded by the code, organized into labeled categories (Alzheimer’s and Healthy), resized to the same dimensions, converted to grayscale, and normalized to ensure consistent analysis. The dataset must be split into training and validation groups so that the model can be tested fairly. The CNN itself must be simple and understandable, consisting of convolutional, pooling, and dense layers, and it must produce a binary output indicating whether an MRI scan more closely resembles Alzheimer’s disease or a healthy brain.

The model will be trained using standard methods, including the Adam optimizer and binary cross-entropy loss, over a small number of training epochs. Accuracy will be tracked during both training and validation. The code must clearly display results, save the trained model for future use, and include comments explaining each step in simple language. Finally, the project must clearly state its limitations, include a disclaimer that it is for educational and research purposes only, and avoid any real-world medical or diagnostic use.

# Language-Detection-and-Translation

## Overview

This project consists of a deep learning-based language detection model and a Streamlit application for real-time language detection and translation. The model is trained using an LSTM-based neural network, and the app provides a user-friendly interface to detect languages and translate text using the Google Translator API.

## Model Details

The language detection model is built using TensorFlow/Keras and employs an LSTM-based neural network. It was trained on a dataset containing multiple languages, filtered and balanced to ensure fair classification.

## Model Performance:

Training Accuracy: 94.78%

Validation Accuracy: 92.69%

Test Accuracy: 92.20%

## Model Architecture:

Embedding Layer

Bidirectional LSTM Layers

Dense Layers with Dropout

Softmax Activation for Multi-class Classification

## Streamlit Application

The Streamlit app provides two main functionalities:

Language Detection: Users can input a sentence, and the model predicts the language with high accuracy.

Text Translation: Users can translate input text into a specified target language using Google Translator.

## How It Works:

The app loads the trained model and necessary preprocessed data (tokenizer, label mappings, etc.).

Users enter text in the provided text area.

The app provides the detected language name.

If translation is selected, users specify the target language, and the text is translated accordingly.


# Catology: Cat Breed Prediction using Neural Networks

## Overview

Catology is a machine learning project designed to predict the breed of a cat based on a set of input features. The project utilizes a custom neural network implemented from scratch using NumPy. The model is trained on a dataset containing various attributes of cats, such as their behavior, physical characteristics, and environmental factors. The goal is to accurately classify the breed of a cat based on these features.

## Features

- **Custom Neural Network**: A neural network implemented from scratch using NumPy, with support for dropout and L2 regularization.
- **Data Preprocessing**: The dataset is preprocessed to handle missing values, normalize features, and encode categorical variables.
- **Word2Vec Integration**: The project integrates Word2Vec embeddings to handle textual descriptions of cats, allowing for more flexible input.
- **Interactive Prediction**: Users can input a description of a cat, and the model will predict the breed based on the provided information.
- **Visualization**: The project includes optional visualization of training loss and accuracy over epochs.

## Requirements

To run this project, you need the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `gensim`
- `nltk`


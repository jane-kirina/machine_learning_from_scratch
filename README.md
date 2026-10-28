# Machine Learning Algorithms From Scratch

## Introductions

This project contains **from scratch implementations** of core **machine learning algorithms** without using high-level ML libraries such as `scikit-learn`, `TensorFlow`, or `PyTorch`. Each model has comparisons with the results using `scikit-learn`

The goal is to build NumPy-based mini-ML library to understand the mathematical and computational foundations of classical ML algorithms, then validated each implementation against sklearn and toy analytical cases

### Motivation
The primary objective is to gain understanding of **model internals**, **optimization techniques**, and  the **mathematical intuition** and internal mechanics behind popular machine learning models

### Learning Objectives
- Understand ML algorithms mathematically
- Learn optimization techniques
- Build models without libraries
- Strengthen problem-solving skills

### Libraries
- `numpy` - numerical computations
- `pandas` - data handling
- `matplotlib` & `plotly.express`  - visualization
- `scikit-learn` - comparison

## Algorithms

### Regression:

- [x]  Linear Regression

### Classification:
- [x] Logistic Regression
- [x] K-Nearest Neighbors (KNN)
- [ ] Decision Tree
- [ ] Support Vector Machine (SVM)

### Util functions
- [X] Mean Absolute Error, Mean Squared Error, R^2 score
- [X] Accuracy, Precision, Recall, F Beta Score
- [X] ROC AUC
- [] .

### Util functions
- [x] train_test_split


---
## Implementation Details

### General Structure

Each model has the following structure:

1. Introduction
2. Math behind model
3. Implementaion of model
4. Model Training, Prediction and Data Visualization
5. Comparison with `scikit-learn` version

### Project rules
- no sklearn inside algorithm implementation
- same API for every model: fit, predict
- each algorithm has:
  - math explanation
  - implementation
  - test on toy data
  - comparison with sklearn
  - discussion of complexity and failure modes
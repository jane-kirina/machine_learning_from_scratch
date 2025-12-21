# Machine Learning Algorithms From Scratch

## Introductions

This project contains **from scratch implementations** of core **machine learning algorithms** without using high-level ML libraries such as `scikit-learn`, `TensorFlow`, or `PyTorch`. Each model has comparisons with the results using `scikit-learn`

### Motivation
The primary objective is to gain understanding of **model internals**, **optimization techniques**, and  the **mathematical intuition** and internal mechanics behind popular machine learning models

### Learning Objectives
* Understand ML algorithms mathematically
* Learn optimization techniques
* Build models without libraries
* Strengthen problem-solving skills

### Libraries
* `numpy` - numerical computations
* `pandas` - data handling
* `matplotlib` & `plotly.express`  - visualization
* `scikit-learn` - comparison

## Algorithms

### Regression:

- [x]  Linear Regression

### Classification:
- [ ] Logistic Regression
- [ ] K-Nearest Neighbors (KNN)
- [ ] Decision Tree
- [ ] Support Vector Machine (SVM)

---

## Implementation Details

### General Structure

Each model has the following structure:

1. Introduction
2. Math behind model
3. Implementaion of model
4. Model Training, Prediction and Data Visualization
5. Comparison with `scikit-learn` version

### ✔ Linear Regression

- Hypothesis: `y = wx + b`
- Gradient Descent
- Cost function: Mean Squared Error, Mean Squared Error and R² score
- Regularization: Lasso(L1), Ridge(L2)

### ➖ Logistic Regression

- Sigmoid function for probability estimation
- Binary classification
- Loss: Binary Cross-Entropy

### ➖ K-Nearest Neighbors (KNN)

- Lazy learning algorithm
- Classifies based on majority vote of `k` nearest samples
- Distance metric: Euclidean distance

### ➖ Decision Tree

- Recursive tree construction
- Recursive splitting of data
- Criteria: Information Gain / Gini Index
- Handles categorical and numerical data

### ➖ Support Vector Machine (SVM)

- Hard-margin / Soft-margin (basic)
- Hinge loss optimization
- Maximizes margin between classes
- Uses hinge loss
- Linear kernel implementation

## ⭕ Future Enhancements

- Multiclass logistic regression
- Multivariate linear regression
- Pruning in decision trees
- Cross-validation support


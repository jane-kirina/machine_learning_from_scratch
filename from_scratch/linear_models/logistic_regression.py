# import
import numpy as np


# Implementation of Logistic Regression
class SimpleLogisticRegression:
    '''
    Logistic Regression implemented using Gradient Descent
    Supports optional L1 (Lasso) and L2 (Ridge) regularization

    Parameters
    ----------
    learning_rate : float, default=0.001
        Step size used to update model parameters during gradient descent
    epochs : int, default=1000
        Number of iterations over the training dataset
    penalty : {'l1', 'l2', None}, default=None
        Type of regularization to apply
    alpha : float, default=0.0
        Regularization strength
    '''
    def __init__(self, learning_rate=0.01, epochs=1000, penalty=None, alpha=0.0):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.penalty = penalty
        self.alpha = alpha

        # Model parameters (initialized during training)
        self.weight = None
        self.bias = None

    def _sigmoid(self, z):
        '''
        Sigmoid activation function
        '''
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        '''
        Train the Logistic regression model using gradient descent

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix
        y : ndarray of shape (n_samples,)
            Target values
        '''
        # Init parameters
        # Number of samples and features
        n_samples, n_features = X.shape

        # Initialize weights and bias to zero
        self.weight = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent loop
        for _ in range(self.epochs):
            # 1. Linear combination
            linear_output = np.dot(X, self.weight) + self.bias

            # 2. Apply sigmoid to get probabilities
            y_predicted = self._sigmoid(linear_output)

            # 3. Compute gradients (log-loss derivatives)
            error = y_predicted - y

            bias_derivative = (1 / n_samples) * np.sum(error)
            weight_derivative = (1 / n_samples) * np.dot(X.T, error)

            # 4. Add regularization for derivative if enabled
            if self.penalty == 'l1':
                # L1 adds absolute value penalty (sparse weights)
                weight_derivative += (self.alpha / n_samples) * np.sign(self.weight)
            elif self.penalty == 'l2':
                # L2 adds squared penalty (smooth weights)
                weight_derivative += (self.alpha / n_samples) * self.weight

            # 5. Update model parameters
            self.bias -= self.learning_rate * bias_derivative
            self.weight -= self.learning_rate * weight_derivative

    def predict_proba(self, X):
        '''
        Compute predicted probabilities for the positive class (y = 1)

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted probabilities for class 1, with values in the range [0, 1]

        '''
        # sigmoid(X @ weights + bias)
        linear_output = np.dot(X, self.weight) + self.bias
        return self._sigmoid(linear_output)

    def predict(self, X, threshold=0.5):
        '''
        Predict binary class labels for input samples

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix

        threshold : float, default=0.5
            Decision threshold used to convert probabilities into class labels
            - If probability >= threshold → class 1
            - Otherwise → class 0

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted class labels (0 or 1)
        '''
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def get_params(self):
        '''
        Retrieve model parameters and hyperparameters

        Returns
        -------
        dict
            Dictionary containing weights, bias, and training settings
        '''
        return {
            'weights': self.weight,
            'bias': self.bias,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'penalty': self.penalty
        }
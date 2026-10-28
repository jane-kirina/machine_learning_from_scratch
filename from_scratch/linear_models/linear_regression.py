# import
import numpy as np

# Implementation of Simple Linear Regression
class SimpleLinearRegression:
    '''
    Simple Linear Regression implemented using Gradient Descent
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

    def __init__(self, learning_rate=0.01, epochs=2000, penalty=None, alpha=0.0):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.penalty = penalty
        self.alpha = alpha

        # Model parameters (initialized during training)
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        '''
        Train the linear regression model using gradient descent

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
            # 1. Calculate linear equation/prediction
            # y_hat = Xw + b
            y_predicted = np.dot(X, self.weight) + self.bias

            # 2. Computes the gradient
            # If gradient is positive, the parameter is too high -> decrease
            # If gradient is negative, the parameter is too low -> increase
            bias_derivative = (1 / n_samples) * np.sum(y_predicted - y)
            weight_derivative = (1 / n_samples) * np.dot(X.T, (y_predicted - y))

            # 3. Add regularization for derivative if enabled
            if self.penalty == 'l1':
                # L1 adds absolute value penalty (sparse weights)
                weight_derivative += (self.alpha / n_samples) * np.sign(self.weight)
            elif self.penalty == 'l2':
                # L2 adds squared penalty (smooth weights)
                weight_derivative += (self.alpha / n_samples) * self.weight

            # 4. Update model parameters
            # If the gradient is positive -> decrease the parameter
            # If the gradient is negative -> increase the parameter
            self.bias -= self.learning_rate * bias_derivative
            self.weight -= self.learning_rate * weight_derivative

    def predict(self, X):
        '''
        Predict target values using the trained model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        ndarray
            Predicted values
        '''
        return np.dot(np.array(X), self.weight) + self.bias

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

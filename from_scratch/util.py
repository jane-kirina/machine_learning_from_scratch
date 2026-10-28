#Imports
import numpy as np
import random

def train_test_split(X, y=None, test_size=0.3, random_state=None, shuffle=True):
    '''
        Split features and labels into training and testing sets
    
        This function is a simple implementation of basic behavior of sklearn.model_selection.train_test_split
        It supports optional shuffling and reproducible splits using a random seed
    '''
    # Total number of samples
    n = len(X)
    
    # Number of samples to use for training
    train_size = int(n * (1 - test_size))

    # Create a list of all row positions
    indices = list(range(n))
    
    # Set random seed for reproducibility
    if random_state is not None:
        random.seed(random_state)

    # Shuffle indices if True
    if shuffle:
        random.shuffle(indices)
        
    # Split indices into training and testing sets
    training_sample = indices[:train_size]
    testing_sample = indices[train_size:]
    
    if y is None:
        return (
            select_rows(X, training_sample),
            select_rows(X, testing_sample),
        )

    # Select rows
    x_train = select_rows(X, training_sample)
    x_test = select_rows(X, testing_sample)

    y_train = select_rows(y, training_sample)
    y_test = select_rows(y, testing_sample)

    return x_train, x_test, y_train, y_test


def select_rows(data, indices):
    if hasattr(data, "iloc"):
        return data.iloc[indices] # Pandas

    return data[indices] # NumPy
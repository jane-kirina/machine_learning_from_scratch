#Imports
import numpy as np
import random

# Implementation of train_test_split from scikit-learn
def train_test_split(X, y=None, test_size=0.3, random_state=None, shuffle=True):
    '''
        Split features and labels into training and testing sets
    
        This function is a simple implementation of basic behavior of sklearn.model_selection.train_test_split
        It supports optional shuffling and reproducible splits using a random seed
    
        Parameters
        ----------
        x : pandas.DataFrame or pandas.Series
            Feature dataset to be split
        y : pandas.Series or pandas.DataFrame
            Target labels corresponding to x
        test_size : float, default=0.3
            Proportion of the dataset to include in the test split (between 0 and 1)
        random_state : int or None, default=None
            Seed for the random number generator to ensure reproducibility
            If None, results will be different each run
        shuffle : bool, default=True
            True - randomly shuffle before splitting
            False - keep original order
    
        Returns
        -------
        x_train : same type as x
            Training portion of the features
        x_test : same type as x
            Testing portion of the features
        y_train : same type as y
            Training portion of the labels
        y_test : same type as y
            Testing portion of the labels
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
        return X.iloc[training_sample], X.iloc[testing_sample]


    # Select rows using positional indexing
    x_train = X.iloc[training_sample]
    y_train = y.iloc[training_sample]
    x_test = X.iloc[testing_sample]
    y_test = y.iloc[testing_sample]

    return x_train, x_test, y_train, y_test
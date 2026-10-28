import numpy as np

def euclidean_distance(x, z):
    return np.sqrt(np.sum((x - z) ** 2))

def hamming_distance(x, z):
    x = np.asarray(x)
    z = np.asarray(z)
    return np.sum(x != z)

def normalized_hamming_distance(x, z):
    x = np.asarray(x)
    z = np.asarray(z)
    return np.mean(x != z)

def minkowski_distance(x, z, p=2):
    x = np.asarray(x)
    z = np.asarray(z)
    return np.sum(np.abs(x - z) ** p) ** (1 / p)

def manhattan_distance(x, z):
    x = np.asarray(x)
    z = np.asarray(z)
    return np.sum(np.abs(x - z))

class SimpleKNN:
    def __init__(self, k = 3, regression=False, distance = euclidean_distance):
        self.k = k
        self.regression=regression
        self.distance = distance
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def prediction(self, x_test_i):
        distances = np.array([
            self.distance(x_test_i, x_train_i)
            for x_train_i in self.X_train
        ])

        k_nearest_indexes = np.argsort(distances)[:self.k]
        targets = self.y_train[k_nearest_indexes]

        return np.mean(targets) if self.regression else np.bincount(targets).argmax()

    def predict(self, X_test):
        return np.array([self.prediction(x) for x in X_test])

    def predict_proba(self, X_test):
        if self.regression:
            raise ValueError("predict_proba is only for classification")
    
        classes = np.unique(self.y_train)
        probabilities = []
    
        for x_test_i in X_test:
            distances = np.array([
                self.distance(x_test_i, x_train_i)
                for x_train_i in self.X_train
            ])
    
            k_nearest_indexes = np.argsort(distances)[:self.k]
            targets = self.y_train[k_nearest_indexes]
    
            class_probabilities = [
                np.mean(targets == label)
                for label in classes
            ]
            probabilities.append(class_probabilities)
    
        return np.array(probabilities)
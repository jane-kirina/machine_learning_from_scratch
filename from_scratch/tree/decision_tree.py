import numpy as np
import math
'''
TODO

DecisionTreeClassifier from scratch:
- binary splits only
- numerical features only
- Gini impurity
- max_depth
- min_samples_split
- min_samples_leaf
- predict
- predict_proba

then add:
- entropy
- regression tree
- categorical features
- pruning
- feature importance

'''

class Node:
    def __init__(self):
        self.split = None # threshold value used to split the data
        self.feature = None # index of the feature used for splitting
        self.left = None
        self.right = None
        self.leaf_value = None # predicted class if this node becomes a leaf

        
    def set_params(self, split: float, feature: int):
        """
        Set the split & feature parameters for this node
        
        Input:
            split   -> value to split feature on
            feature -> index of feature to be used in splitting 
        """
        self.split = split
        self.feature = feature
        
    def get_params(self):
        return(self.__split, self.__feature)
    
    def set_children(self, left, right):
        self.__left  = left
        self.__right = right

    def get_left_node(self):
        return(self.__left)
    
    def get_right_node(self):
        return(self.__right)
    
class DecisionTree():
    def __init__(self, max_depth=100, min_samples_split=2):
        """
        Initializer
        
        Inputs:
            max_depth         -> maximum depth the tree can grow
            min_samples_split -> minimum number of samples required to split a node
        """
        self.tree              = None
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
    
    def _is_finished(self, depth):
        if (depth >= self.max_depth
            or self.n_class_labels == 1
            or self.n_samples < self.min_samples_split):
            return True
        return False
    
    def _entropy(self, y):
        return -sum([p * math.log(p,2) for p in y])

    def _create_split(self, X, thresh):
        pass

    def _information_gain(self, X, y, thresh):
        pass

    def _best_split(self, X, y, features):
        pass
    
    def _build_tree(self, X, y, depth=0):
        pass
    
    def _traverse_tree(self, x, node):
        pass
     
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

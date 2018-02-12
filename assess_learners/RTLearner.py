"""
A simple wrapper for Random Tree Regression. (c) 2018 T. Ruzmetov
"""
import numpy as np
from random import randint
from copy import deepcopy
from collections import Counter
from operator import itemgetter

class RTLearner(object):

    def __init__(self, leaf_size, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = np.array([])
        if verbose:
            self.get_learner_info()
        
    def author(self):
        return 'truzmetov3'

    def get_split_indices(self, dataX, num_samps):
        """
        This function randomly chooses a feature to split the outcome
        based on median value of chosen feature. It returns: 
           left_index - row indecies that go left
           right_index - row indecies that go right
           feature_index - randomly selected feature index
           split_val - median calue of selected feature used to performe splitting
        """
        feature_index = randint(0, dataX.shape[1] - 1)
        split_val = (dataX[randint(0, num_samps -1)][feature_index]
                     + dataX[randint(0,num_samps-1)][feature_index])/2

        left_index = [i for i in xrange(dataX.shape[0])
                        if dataX[i][feature_index] <= split_val]
        right_index = [i for i in xrange(dataX.shape[0])
                         if dataX[i][feature_index] > split_val]
        return left_index, right_index, feature_index, split_val
    

    def build_tree(self, x_train, y_train):
        num_instances = x_train.shape[0]
        if num_instances == 0:
            print 'all -1s'
            return np.array([-1, -1, -1, -1])
        if num_instances <= self.leaf_size:
            #If there's only one instance, take the mean of the labels
            return np.array([-1, np.mean(y_train), -1, -1])

        values = np.unique(y_train)
        if len(values) == 1:
            # If all instances have the same label, return that label
            return np.array([-1, y_train[0], -1, -1])

        # Choose a random feature, and a random split value
        left_indices, right_indices, feature_index, split_val = \
            self.get_split_indices(x_train, num_instances)

        while len(left_indices) < 1 or len(right_indices) < 1:
            left_indices, right_indices, feature_index, split_val = \
                self.get_split_indices(x_train, num_instances)

        left_x_train = np.array([x_train[i] for i in left_indices])
        left_y_train = np.array([y_train[i] for i in left_indices])
        right_x_train = np.array([x_train[i] for i in right_indices])
        right_y_train = np.array([y_train[i] for i in right_indices])

        left_tree = self.build_tree(left_x_train, left_y_train)
        right_tree = self.build_tree(right_x_train, right_y_train)

        if len(left_tree.shape) == 1:
            num_left_side_instances = 2
        else:
            num_left_side_instances = left_tree.shape[0] + 1

        root = [feature_index, split_val, 1, num_left_side_instances]
        return np.vstack((root, np.vstack((left_tree, right_tree))))

    def addEvidence(self, Xtrain, Ytrain):
        self.tree = self.build_tree(Xtrain, Ytrain)

        
    def traverse_tree(self, instance, row=0):
        feature_index = int(self.tree[row][0])
        if feature_index == -1:
            return self.tree[row][1]
        if instance[feature_index] <= self.tree[row][1]:
            return self.traverse_tree(instance, row + int(self.tree[row][2]))
        else:
            return self.traverse_tree(instance, row + int(self.tree[row][3]))

    def query(self, Xtest):
        result = []
        for instance in Xtest:
            result.append(self.traverse_tree(instance))
        return np.array(result)

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"

    

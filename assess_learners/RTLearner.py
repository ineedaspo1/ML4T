"""
A simple wrapper for Random Tree Regression. (c) 2018 T. Ruzmetov
"""
import numpy as np
import pandas as pd
from random import randint
from copy import deepcopy
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

        left_index = [i for i in range(dataX.shape[0])
                        if dataX[i][feature_index] <= split_val]
        right_index = [i for i in range(dataX.shape[0])
                         if dataX[i][feature_index] > split_val]
        return left_index, right_index, feature_index, split_val
    

    def buildTree(self, dataX, dataY):
        """

        """
        
        num_samps = dataX.shape[0]
        num_feats = dataX.shape[1]
        
        ##########################################################
        #return the most common value from the root of current node if no sample left
        if num_samps < 1:
            return np.array([-1, -1, np.nan, np.nan])

        # return leaf, if there are <= leaf_size samples 
        if num_samps <= self.leaf_size:
            return np.array([-1, np.mean(dataY), np.nan, np.nan])

        # return leaf, if all data in dataY are the same
        if len(np.unique(dataY)) == 1:
            return np.array([-1, dataY[0], np.nan, np.nan])
        ##########################################################

        
        # Choose a random feature, and a random split value
        left_indices, right_indices, feature_index, split_val = \
            self.get_split_indices(dataX, num_samps)

        while len(left_indices) < 1 or len(right_indices) < 1:
            left_indices, right_indices, feature_index, split_val = \
                self.get_split_indices(dataX, num_samps)

        # Build left and right branches and the root                    
        lefttree = self.buildTree(dataX[left_indices], dataY[left_indices])
        righttree = self.buildTree(dataX[right_indices], dataY[right_indices])
        
        ######################################################################
        # Set the starting row for the right subtree of the current root
        if lefttree.ndim == 1:
            righttree_start = 2 # The right subtree starts 2 rows down
        elif lefttree.ndim > 1:
            righttree_start = lefttree.shape[0] + 1
        
        ######################################################################
        
        root = [feature_index, split_val, 1, righttree_start]
        return np.vstack((root, lefttree, righttree))


    def recur_search(self, point, row=0):
        feature_index = int(self.tree[row][0])
        if feature_index == -1:
            return self.tree[row][1]
        if point[feature_index] <= self.tree[row][1]:
            return self.recur_search(point, row + int(self.tree[row][2]))
        else:
            return self.recur_search(point, row + int(self.tree[row][3]))

        
    def addEvidence(self, dataX, dataY):
        self.tree = self.buildTree(dataX, dataY)
        if self.verbose:
            self.get_learner_info()
        
   
    def query(self, dataX):
        """ Performe prediction on test set given the model we built
        Params:  dataX -np ndarray of test 
        Returns: preds: 1D np array of the estimated values
        """
        preds = []
        for rows in dataX:
            preds.append(self.recur_search(rows))
        return np.array(preds)
    

    def get_learner_info(self):
        print ("Info about this Random Tree Learner:")
        print ("leaf_size =", self.leaf_size)
        if self.tree is not None:
            print ("tree shape =", self.tree.shape)
            print ("tree as a matrix:")
            # Create a dataframe from tree for a user-friendly view
            df_tree = pd.DataFrame(self.tree, columns=["factor", "split_val", "left", "right"])
            df_tree.index.name = "node"
            print (df_tree)
        else:
            print ("Tree has no data")
            

if __name__=="__main__":
    print "No more secret clues"

    

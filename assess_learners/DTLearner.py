"""
A simple wrapper for Decision Tree Regression. (c) 2018 T. Ruzmetov
"""
import numpy as np
import pandas as pd
from copy import deepcopy
from operator import itemgetter
from collections import Counter

class DTLearner(object):

    def __init__(self, leaf_size=1, verbose=False, tree=None):
       self.leaf_size = leaf_size
       self.verbose = verbose
       self.tree = deepcopy(tree)
       if verbose:
            self.get_learner_info()
            
    def author(self):
        return 'truzmetov3'        

    def buildTree(self, dataX, dataY, rootX=[], rootY=[]):
        """
        This is eager DT algorithm for regression that chooses best feature for splitting based
        on its highest abs(corr(X_i,Y)). Median of the chosen feature is used as splitting value.
        If all features have the same abs(corr(X_i,Y)), choose the first feature and pass.
        If the best feature can't split the target into two groups, choose the next best feature; 
        if none of the features do, return the leaf.
        """

        num_feats = dataX.shape[1]
        num_samps = dataX.shape[0]

        ######################################################################
        if num_samps < 1:
            return np.array([-1, -1, np.nan, np.nan])

        if num_samps <= self.leaf_size:
            return np.array([-1, np.mean(dataY), np.nan, np.nan])

        if len(np.unique(dataY)) == 1:
            return np.array([-1, dataY[0], np.nan, np.nan])
        ######################################################################

            
        remain_feats_for_split = list(range(num_feats))

        # calculate coor(X_i,Y)
        corrs = []
        for i in range(num_feats):
            abs_corr = abs(np.corrcoef(dataX[:,i], dataY)[0,1])
            corrs.append((i, abs_corr))
        
        # Sort corrs in descending order
        corrs = sorted(corrs, key=itemgetter(1), reverse=True)

        feat_corr_i = 0
        while len(remain_feats_for_split) > 0:
            best_feat_i = corrs[feat_corr_i][0]
            best_abs_corr = corrs[feat_corr_i][1]

            # calculate split_val by taking median over best feature
            split_val = np.median(dataX[:, best_feat_i])

            # get boolean indecies for left and right splitting
            left_index = dataX[:, best_feat_i] <= split_val
            right_index = dataX[:, best_feat_i] > split_val

            # break out of the loop if split is successful            
            if len(np.unique(left_index)) > 1:
                break
            
            remain_feats_for_split.remove(best_feat_i)
            feat_corr_i += 1
            
        #If we complete the while loop and run out of features to split, return leaf
        if len(remain_feats_for_split) == 0:
            return np.array([-1, Counter(dataY).most_common(1)[0][0], np.nan, np.nan])

        # Build left and right branches and the root                    
        lefttree = self.buildTree(dataX[left_index], dataY[left_index], dataX, dataY)
        righttree = self.buildTree(dataX[right_index], dataY[right_index], dataX, dataY)

        ##############################################################################
        # Set the starting row for the right subtree of the current root
        if lefttree.ndim == 1:
            righttree_start = 2 # The right subtree starts 2 rows down
        elif lefttree.ndim > 1:
            righttree_start = lefttree.shape[0] + 1
        #############################################################################
        
        root = np.array([best_feat_i, split_val, 1, righttree_start])
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
            preds.append(self.recur_search(rows, row=0))
        return np.asarray(preds)


    def get_learner_info(self):
        print ("Info about this Decision Tree Learner:")
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

"""
A simple wrapper for InsaneLearners.  (c) 2018 T. Ruzmetov
"""
import BagLearner as bl
import LinRegLearner as lrl
import numpy as np

class InsaneLearner(object):
    
    def __init__(self, verbose = False):
        n_repeats = 20
        self.learnerList = []
        self.n_repeats = n_repeats 
        self.verbose = verbose
        for i in range(n_repeats):
            self.learnerList.append(bl.BagLearner(lrl.LinRegLearner,
                                             kwargs = {},
                                             bags = 20,
                                             boost = False,
                                             verbose = self.verbose))

            
    def author(self):
        return 'truzmetov3'    
      
    def addEvidence(self, trainX, trainY):
        for learner in self.learnerList:  
            learner.addEvidence(trainX, trainY)
           
    def query(self, testX):
        pred = np.empty( (testX.shape[0], self.n_repeats) )  
        for col in range(self.n_repeats):
            pred[:,col] = self.learnerList[col].query(testX)
        return pred.mean(axis = 1)

if __name__=="__main__":
    print "Health is much more important than the wealth!"

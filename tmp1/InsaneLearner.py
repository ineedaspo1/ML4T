"""
A simple wrapper for InsaneLearners.  (c) 2018 T. Ruzmetov
"""
import BagLearner as bl
import LinRegLearner as lrl
import numpy as np

class InsaneLearner(object):
    
    def __init__(self, verbose = False):
        learnerList = []
        num = 20 
        self.verbose = verbose
        for i in range(num):
            learnerList.append(bl.BagLearner(lrl.LinRegLearner, kwargs = {}, \
                bags = 20, verbose = self.verbose))
            
        self.learnerList = learnerList
        self.num = num

    def author(self):
        return 'truzmetov3'    
      
    def addEvidence(self, trainX, trainY):
        for learner in self.learnerList:
            learner.addEvidence(trainX, trainY)
           
    def query(self, testX):
        pred = np.empty( (testX.shape[0], self.num) )  
        for col in range(self.num):
            pred[:,col] = self.learnerList[col].query(testX)
            return pred.mean(axis = 1)
        
if __name__=="__main__":
    print " mama mia "

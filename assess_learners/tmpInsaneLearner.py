"""
A simple wrapper for InsaneLearner.  (c) 2018 T. Ruzmetov
"""
import numpy as np
import LinRegLearner as lrl
import BagLearner as bl

class InsaneLearner(object):

    def __init__(self,verbose = False):
        self.learners = []
        self.repeats = 20
        
    def author(self):
        return 'truzmetov3'    
             
    def addEvidence(self, dataX, dataY):

        for i in range(0, self.repeats):
            learner = bl.BagLearner(learner = lrl.LinRegLearner,
                                    kwargs = {},
                                    bags = 20,
                                    boost = False,
                                    verbose = False)
            self.learners[i].learner.addEvidence(dataX, dataY)
               
    def query(self, points):
        y_pred = np.empty([self.repeats, points.shape[0]])
        for i in range(0, self.repeats):
            y_pred[i] = self.learners[i].query(points)
        return np.mean(y_pred, axis=0)
    
if __name__=="__main__":
    print " Mama mia "

"""
A simple wrapper for Bagging.  (c) 2018 T. Ruzmetov
"""
import numpy as np

class BagLearner(object):

    def __init__(self, learner, kwargs, bags, boost, verbose):
        self.learner = learner(**kwargs)   
        self.boost = boost
        self.bags = bags
        self.learners = []
        for i in range(0,self.bags):
            self.learners.append(learner(**kwargs))
    
    def author(self):
        return 'truzmetov3'    
             
    def addEvidence(self, dataX, dataY):
        """ Add training data to learner """
        
     #   if self.boost:
     #       sample_index = np.random.choice(len(dataY), len(dataY), replace=True)
     #       for i in range(0,self.bags):
     #           dataXsamp = dataX[sample_index,:]            
     #           dataYsamp = dataY[sample_index]

     #           adaboost = self.learner
     #           adaboost.addEvidence(dataXsamp, dataYsamp)
     #           Yada = adaboost.query(dataX)
     #           error = abs(dataY - Yada)/sum(abs(dataY - Yada))
     #           sample_index = np.random.choice(len(dataY), len(dataY), replace=True, p = error) 
     #           self.learners[i].addEvidence(dataXsamp, dataYsamp)            
     #   else:   

        for i in range(0,self.bags):
            sample_size = len(dataY)
            sample_index = np.random.choice(sample_size, sample_size, replace=True)
            Xsamp = dataX[sample_index,:]            
            Ysamp = dataY[sample_index]
            self.learners[i].addEvidence(Xsamp, Ysamp)
            
            
    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        y_pred = np.empty([self.bags, points.shape[0]])
        for i in range(0,self.bags):
            y_pred[i] = self.learners[i].query(points)
        return np.mean(y_pred, axis=0)
    
if __name__=="__main__":
    print " mama mia "

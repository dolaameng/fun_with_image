from PIL import Image
from pylab import *
from numpy.random import randn

class KnnClassifier(object):
    def __init__(self, labels, samples):
        """Initialize classifier with training data"""
        self.labels = labels
        self.samples = samples
        
    def classify(self, point, k = 3):
        """Classify a point against k nearest in 
        the training data, return label"""
        ## distance to all training points
        dist = array([KnnClassifier.l2dist(point, s) for s in self.samples])
        ## sort them
        ndx = dist.argsort()
        ## use dictionary to store the k nearest
        votes = {}
        for i in range(k):
            label = self.labels[ndx[i]]
            votes.setdefault(label, 0)
            votes[label] += 1
        return max(votes)
    @staticmethod     
    def l2dist(p1, p2):
        return sqrt(sum( (p1-p2) ** 2 ))
        
def test():
    ## test knn with simple random
    """
    n = 200
    class1 = 0.6 * randn(n, 2)
    class2 = 1.2 * randn(n, 2) + array([5, 1])
    labels = hstack((ones(n), -ones(n)))
    model = KnnClassifier(labels, vstack((class1, class2)))
    predictions = [model.classify(point) for point in vstack((class1, class2))]
    print 'accuracy is :', sum(labels == predictions) * 100 / len(labels), '%'
    """
    show()
    
    
if __name__ == '__main__':
    test()
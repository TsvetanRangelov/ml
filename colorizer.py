import numpy as np
class Perceptron(object):
    def __init__(self, data, classification):
        self.data = np.hstack((np.ones((len(data),1)),data))
        self.weights = np.random.uniform(0,1,len(self.data[0]))
    def eval(self, point):
        value = np.dot(self.weights, point)
        if(value>0.0):
            return 1.0
        else:
            return -1.0

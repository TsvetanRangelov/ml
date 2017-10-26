import numpy as np
class Perceptron(object):
    def __init__(self, dimension):
        self.weights = np.random.uniform(0,1,dimension)
    def eval(self, point):
        return np.tanh(np.dot(self.weights, point))

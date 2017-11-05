import numpy as np
class Perceptron(object):
    def __init__(self, dimension):
        self.weights = np.random.uniform(-1,1,dimension)
    def eval(self, point):
        return np.tanh(self.doteval(point))
    def doteval(self, point):
        return np.dot(self.weights, point)
    def __repr__(self):
        return str(self.weights)
    def update_delta(self, point, y, deltas=[], weights=[], last=False):
        if(last):
            self.delta = (1-self.eval(point)**2)*2*(self.eval(point)-y)
        else:
            self.delta = (1-self.eval(point)**2)

import numpy as np

class Regression(object):
    def __init__(self, data):
        self.x = np.array(list(map(lambda data: np.append(1,data[:-1]), data)))
        self.y = np.array([x[len(x)-1] for x in data])
    def get_pseudo_inverse(self):
        return np.dot(np.linalg.inv(np.dot(self.x.T,self.x)), self.x.T)
    def calc_weight(self):
        self.weights = np.dot(self.get_pseudo_inverse(), self.y)
    def get_weight_vector(self):
        return self.weights
    def predict(self, point):
        return np.dot(self.weights, np.append(1,point))

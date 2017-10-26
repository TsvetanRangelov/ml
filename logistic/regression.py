import numpy as np

class Regression(object):
    def __init__(self, data):
        self.x = np.array(list(map(lambda data: np.append(1,data[:-1]), data)))
        self.y = np.array([x[len(x)-1] for x in data])
        self.weights = np.random.uniform(-1,1, len(self.x[0]))
        self.etha = 0.1
        self.learn_weights(1000)
    def get_error_point(self, index):
        return (self.y[index]*self.x[index]) / (1 + np.exp(self.y[index]*np.dot(self.weights.transpose(), self.x[index])))
    def get_error_gradient(self):
        error_sum = np.zeros(len(self.x[0]));
        for i in range(len(self.x)):
            error_sum += self.get_error_point(i)
        return error_sum/len(self.x)
    def learn_weights(self, time):
        for i in range(time):
            self.weights = self.weights + self.etha*self.get_error_gradient()
    def get_weight_vector(self):
        return self.weights
    def classify(self, x):
        return np.tanh(np.dot(self.weights, x))

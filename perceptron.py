import numpy as np
class Perceptron(object):
    def __init__(self, data, classification):
        self.data = np.hstack((np.ones((len(data),1)),data))
        self.weights = np.random.uniform(0,1,len(self.data[0]))
        for i in range(100):
            missclassed=0
            for i in range(len(self.data)-1):
                if(self.eval(self.data[i]) != classification[i]):
                    missclassed+=1
                    self.weights = np.add(self.weights,classification[i]*self.data[i])
            if(missclassed==0):
                break
    def eval(self, point):
        value = np.dot(self.weights, point)
        if(value>0.0):
            return 1.0
        else:
            return -1.0

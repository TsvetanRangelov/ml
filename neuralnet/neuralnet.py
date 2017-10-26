import numpy as np
from perceptron import Perceptron
class Neuralnet(object):
    def __init__(self, data, layers=[4,3,2,1]):
        self.layers = layers;
        self.x = np.array(list(map(lambda data: np.append(1,data[:-1]), data)))
        self.y = np.array([x[len(x)-1] for x in data])
        self.perceptrons = [[Perceptron(([len(self.x[0])-1]+layers)[i]+1) for u in range(layers[i])] for i in range(len(layers))]
        for i in range(100):
            self.step()
    def step(self):
        index = np.random.randint(len(self.x))
        inputs = []
        inputs.append(self.x[index])
        for i in range(len(self.layers)):
            inputs.append(np.append(1,[p.eval(inputs[i]) for p in self.perceptrons[i]]))
        print(inputs[len(self.layers)][1])

import numpy as np
from perceptron import Perceptron
class Neuralnet(object):
    def __init__(self, data, layers=[4,1]):
        self.layers = layers
        self.eta = 0.05
        self.x = np.array(list(map(lambda data: np.append(1,data[:-1]), data)))
        self.y = np.array([x[len(x)-1] for x in data])
        self.perceptrons = [[Perceptron(([len(self.x[0])-1]+layers)[i]+1) for u in range(layers[i])] for i in range(len(layers))]
        for i in range(10000):
            index = np.random.randint(len(self.x))
            inputs = self.feed_forward(index)
            deltas = self.backpropagation(inputs, index)
            self.update_weights(inputs, deltas)
    def update_weights(self, inputs, deltas):
        for i in range(len(self.layers)):
            for j in range(len(self.perceptrons[i])):
                self.perceptrons[i][j].weights -= self.eta*inputs[i]*deltas[i][j]
    def feed_forward(self, index):
        inputs = []
        inputs.append(self.x[index])
        for i in range(len(self.layers)):
            inputs.append(np.append(1,[p.eval(inputs[i]) for p in self.perceptrons[i]]))
        return inputs
    def total_error(self):
        sum=0
        for i in range(len(self.x)):
            sum+=self.error_on(i)
        return sum/len(self.x)
    def error_on(self, index):
        return abs(self.feed_forward(index)[len(self.layers)][1]-self.y[index])

    def backpropagation(self, inputs, index):
        deltas = []
        row = []
        for perc in self.perceptrons[len(self.layers)-1]:
            row.insert(0, (1-perc.eval(inputs[len(inputs)-2])**2)*2*(perc.eval(inputs[len(inputs)-2])-self.y[index]))
        deltas.append(row)
        for i in reversed(range(len(self.layers)-1)):
            row = []
            for j in range(len(self.perceptrons[i])):
                weights = [per.weights[j+1] for per in self.perceptrons[i+1]]
                row.append((1-self.perceptrons[i][j].eval(inputs[i])**2)*np.dot(deltas[0],weights))
            deltas.insert(0,row)
        return deltas

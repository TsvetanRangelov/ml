import matplotlib.pyplot as pyplot
import pandas
from perceptron import Perceptron #We are importing the module here

data = pandas.read_csv('data.csv').values
columns = data.transpose()
pyplot.scatter(columns[0], columns[1], c=columns[2])
perceptron = Perceptron(columns[:-1].transpose(), columns[2]) #passing the data appropriately

w = perceptron.weights #getting the final weights

pyplot.plot([-1, 1], [(w[1]-w[0])/w[2],(-w[1]-w[0])/w[2]]) # drawing our prediction
pyplot.show()

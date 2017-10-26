import pandas
import matplotlib.pyplot as pyplot
from neuralnet import Neuralnet

data = pandas.read_csv('data.csv').values
columns = data.transpose()
n=Neuralnet(data)

pyplot.scatter(columns[0], columns[1])
pyplot.show()

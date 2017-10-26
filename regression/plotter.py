import pandas
import matplotlib.pyplot as pyplot
from regression import *
data = pandas.read_csv('data.csv').values
columns = data.transpose()
r=Regression(data)
line = r.get_weight_vector()
pyplot.plot([-1,1],[line[0]-line[1], line[0]+line[1]])
pyplot.scatter(columns[0], columns[1])
pyplot.show()

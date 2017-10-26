import pandas
import matplotlib.pyplot as pyplot
from regression import *
data = pandas.read_csv('data.csv').values
r=Regression(data)
w = r.get_weight_vector()
pyplot.plot([-1, 1], [(w[1]-w[0])/w[2],(-w[1]-w[0])/w[2]]) # drawing our prediction
for i in range(len(data)):
    if(abs(r.classify(np.append(1,data[i][:-1])))<0.05):
        data[i][len(data[i])-1]=2
columns = data.transpose()
pyplot.scatter(columns[0], columns[1], c=columns[2])
pyplot.show()

import numpy as np
def eval(point, line):
    if(line[0]*point[0]+line[1]*point[1]+line[2]>0.0):
        return 1.0
    else:
        return -1.0
line = np.random.uniform(-1,1,(3))
data = np.random.uniform(-1,1,(100,2))
classified = [np.append(point,eval(point, line)) for point in data]
np.savetxt("data.csv", classified, fmt="%.2f", delimiter=",", header="x0,x1,y", comments="")

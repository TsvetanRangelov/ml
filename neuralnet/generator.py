import numpy as np
data = np.random.uniform(-1, 1,(100,2))
rand = np.random.uniform(-10, 10)
lin_data = [np.append(point[0], point[1]+point[0]*rand) for point in data]
np.savetxt("data.csv", lin_data, header="x0,x1", delimiter=",", comments="", fmt="%.2f")

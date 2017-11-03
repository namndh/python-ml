import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt




path = os.getcwd() + '/ex1data2.txt'
data = pd.read_csv(path, header=None, names=['Size', 'Bedroom', 'Price'] )

print data.head()
print data.describe()

dt = np.loadtxt('ex1data2_1.txt')

# Standardization Feature Normalization

data = (data - data.mean())/data.std()


def compute_cost(x, y, theta, size):
    inner = np.power(((x * theta.T) - y), 2)
    return np.sum(inner) / (2*size)


def gradient_descend(x, y, theta, alpha, iters, size):
    temp = np.matrix(np.zeros(theta.shape))
    param = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (x * theta.T) - y

        for j in range(param):
            term = np.multiply(error,X[:, j])
            temp[0,j] = theta[0,j] - alpha * 1/size * np.sum(term)

        theta = temp
        cost[i] = compute_cost(x,y,theta,size)

    return theta, cost


data.insert(0, 'Ones', 1)

cols = data.shape[1]
size = data.shape[0]

X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols-1:cols]

X = np.matrix(X.values)
y = np.matrix(y.values)

theta = np.matrix(np.array([0, 0, 0]))

print "Initialization Cost Function Value: {}".format(compute_cost(X, y, theta, size))

alpha = 0.03
iters = 1000

g, cost = gradient_descend(X, y, theta, alpha, iters, size)

print "Optimized Cost Function Value: {}".format(compute_cost(X, y, g, size))

plt.plot(np.arange(iters), cost, 'r')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Error vs. Training Epoch')
plt.grid()
plt.show()

from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
xdata = dt[:, 0].T
ydata = dt[:, 1].T
zdata = dt[:, 2].T
print xdata.shape
print ydata.shape
print zdata.shape
ax.scatter3D([xdata], [ydata], [zdata], c=[zdata], cmap = 'Dark2')

plt.show()

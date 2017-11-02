import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = os.getcwd() + '/ex1data1_1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

print data.head()

print data.describe()

dt = np.loadtxt('ex1data1.txt')


X = dt[:, 0]
y = dt[:, 1]


plt.scatter(X, y, edgecolors='black')

plt.xlabel('Population')
plt.ylabel('Profit')
plt.grid()
plt.show()


def compute_cost(x, y, theta):
        inner = np.power(((x*theta.T) - y), 2)
        return np.sum(inner)/(2*len(x))


data.insert(0, 'Ones', 1)

cols = data.shape[1]

X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]


X = np.matrix(X.values)
y = np.matrix(y.values)

theta = np.matrix(np.array([0, 0]))

print "Initialization Cost Function Value: {}".format(compute_cost(X, y, theta))


def gradient_descent(x, y, theta, alpha, iters):
        temp = np.matrix(np.zeros(theta.shape))
        parameters = int(theta.ravel().shape[1])
        cost = np.zeros(iters)

        for i in range(iters):
                error = (X * theta.T) - y

                for j in range(parameters):
                        term = np.multiply(error, X[:, j])
                        temp[0, j] = theta[0, j] - alpha * 1/(len(X)) * np.sum(term)
                theta = temp
                cost[i] = compute_cost(x, y, theta)
        return theta, cost


alpha = 0.01
iters = 1000


g, cost = gradient_descent(X, y, theta, alpha, iters)

print 'Optimized Cost Function Value: {}'.format(compute_cost(X, y, g))


x = X[:, 1]
f = g[0, 0] + (g[0, 1] * x)


plt.plot(x, f, 'r', label='Prediction')
plt.scatter(dt[:, 0], dt[:, 1], edgecolors='black')
plt.legend(loc=2)
plt.xlabel('Population')
plt.ylabel('Profit')
plt.title('Predicted Profit vs Population Size')

plt.grid()
plt.show()


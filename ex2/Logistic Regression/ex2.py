import os
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

path = os.getcwd() + '/ex2data1_1.txt'
data = pd.read_csv(path, header=None, names=['Exam1', 'Exam2', 'Admitted'])

print data.head()
print data.describe()

positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Exam1'], positive['Exam2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam1'], negative['Exam2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
plt.xlabel('Exam1')
plt.ylabel('Exam2')
plt.grid()
plt.show()


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))


data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]

# convert to numpy arrays and initialize the parameter array theta
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)

print 'Initialization Cost Function Value: {}'.format(cost(theta, X, y))


def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)

    return grad


result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))

print 'Optimized Cost Function Value: {}'.format(cost(result[0], X, y))


def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


theta_min = np.matrix(result[0])
X = np.matrix(X)
print theta_min.shape
prediction = predict(theta_min, X)


correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a,b) in zip(prediction, y)]


accuracy = (sum(map(int, correct)) % len(correct))


print 'Accuracy = {}%'.format(accuracy)

x1 = X[:, 1]
# y = X[:, 2]
f = (theta_min[:, 0] + (x1*theta_min[:, 1]))/(-theta_min[:, 2])


fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Exam1'], positive['Exam2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam1'], negative['Exam2'], s=50, c='r', marker='x', label='Not Admitted')
plt.xlabel('Exam1')
plt.ylabel('Exam2')
ax.plot(x1, f, 'r', label='Decison Boundary')
ax.legend()
plt.grid()
plt.show()

import os
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

path = os.getcwd() + '/ex2data2.txt'
data = pd.read_csv(path, header=None, names=['Test1', 'Test2', 'Accepted'])

positive = data[data['Accepted'].isin([1])]
negative = data[data['Accepted'].isin([0])]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Test1'], positive['Test2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['Test1'], negative['Test2'], s=50, c='r', marker='x', label='Not Accepted')
ax.legend()
plt.xlabel('Test 1')
plt.ylabel('Test 2')
plt.show()

degree = 5
x1 = data['Test1']
x2 = data['Test2']

data.insert(3, 'Ones', 1)

for i in range(1, degree):
    for j in range(0, i):
        data['F' + str(i) + str(j)] = np.power(x1, i - j)*np.power(x2, j)

data.drop('Test1', axis=1, inplace=True)
data.drop('Test2', axis=1, inplace=True)

print data.head()
print data.describe()


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def cost_reg(theta, X, y, lamda):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))

    reg = (lamda / 2 * len(X)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))

    return np.sum(first - second)/len(X) + reg


def grad_reg(theta, X, y, lamda):

    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    param = int(theta.ravel().shape[1])
    grad = np.zeros(param)

    error_func = sigmoid(X * theta.T) - y

    for i in range(param):
        term = np.multiply(error_func, X[:, i])

        if  i == 0:
            grad[i] = np.sum(term) / len(X)

        else:
            grad[i] = (np.sum(term) / len(X)) + ((lamda/len(X)) * theta[:, i])

    return grad


cols = data.shape[1]

X = data.iloc[:, 1:cols]
y = data.iloc[:, 0:1]

X = np.array(X.values)
y = np.array(y.values)

theta = np.zeros(11)

lamda = 1

print "Initialization Cost Function with Regulalization value: {}".format(cost_reg(theta, X, y, lamda))

result = opt.fmin_tnc(func=cost_reg, x0=theta, fprime=grad_reg, args=(X, y, lamda))

theta_min = np.matrix(result[0])

print "Optimized Cost Function with Regulalization value: {}".format(cost_reg(theta_min, X, y, lamda))


def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


prediction = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(prediction, y)]

accuracy = (sum(map(int, correct)) % len(correct))

print "Accuracy of prediction after optimizing: {}%".format(accuracy)






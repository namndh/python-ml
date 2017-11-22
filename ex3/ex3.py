import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cg
from scipy.optimize import minimize
from scipy.io import loadmat
# %matplotlib inline

data = loadmat('ex3data1.mat')

print data['X'].shape
print data['y'].shape


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))

    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))

    return np.sum(first - second)/len(X) + reg


def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    error = sigmoid(X * theta.T) - y

    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)

    # intercept gradient is not regularized
    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)

    return np.array(grad).ravel()


def one_vs_all_fmincg(X, y, num_labels, learning_rate):
    rows = X.shape[0]
    params = X.shape[1]

    # k X (n + 1) array for the parameters of each of the k classifiers
    all_theta = np.zeros((num_labels, params + 1))

    # insert a column of ones at the beginning for the intercept term
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # labels are 1-indexed instead of 0-indexed
    for i in range(1, num_labels + 1):
        theta = all_theta[i - 1, :]
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        # minimize the objective function
        # fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient)
        fmin = fmin_cg(f=cost, x0=theta, fprime=gradient, args=(X, y_i, learning_rate), maxiter=200)
        all_theta[i - 1, :] = fmin

    return all_theta


def one_vs_all_tnc(X, y, num_labels, learning_rate):
    rows = X.shape[0]
    params = X.shape[1]

    all_theta = np.zeros((num_labels, params + 1))

    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    for i in range(1, num_labels + 1):
        theta = all_theta[i - 1, :]
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient, options={'maxiter':200})
        all_theta[i - 1, :] = fmin.x

    return all_theta

print data_test_y.shape

all_theta = one_vs_all_tnc(data['X'], data['y'], 10, 1)
# print all_theta
print all_theta.shape


def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]

    # same as before, insert ones to match the shape
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # convert to matrices
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)

    # compute the class probability for each class on each training instance
    h = sigmoid(X * all_theta.T)

    # create array of the index with the maximum probability
    h_argmax = np.argmax(h, axis=1)

    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1

    return h_argmax


y_pred = predict_all(data['X'][4900:5000, :], all_theta)
print y_pred.shape
# print data['y'][3500:5000, :].shape
# print data['y'][4990:5000, :]
# y_pred = predict_all(data_test_x, all_theta)
# print y_pred.shape
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'][4900:5000, :])]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print 'accuracy = {0}%'.format(accuracy * 100)
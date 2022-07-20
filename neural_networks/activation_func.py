import numpy as np


def sigmoid(x):
    try:
        return 1 / (1 + np.exp(-x))
    except TypeError:
        raise TypeError('A numpy array expected for x.')


def d_sigmoid(x):
    x = sigmoid(x)
    return x * (1 - x)


def d_tanh(x):
    try:
        return 1 - np.tanh(x)**2
    except TypeError:
        raise TypeError('A numpy array expected for x.')


def relu(x):
    try:
        return np.where(x < 0, 0., x)
    except TypeError:
        raise TypeError('A numpy array expected for x.')


def d_relu(x):
    try:
        return np.float64(x > 0)  # Faster than np.where(x > 0, 1., 0.)
    except TypeError:
        raise TypeError('A numpy array expected for x.')


def leaky_relu(x):
    try:
        return np.where(x < 0, 0.01*x, x)
    except TypeError:
        raise TypeError('A numpy array expected for x.')


def d_leaky_relu(x):
    try:
        return np.where(x < 0, 0.01, 1.)
    except TypeError:
        raise TypeError('A numpy array expected for x.')


def softmax(x):
    try:
        exp_scores = np.exp(x)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)
    except TypeError:
        raise TypeError('A numpy array expected for x.')


def d_softmax(x):
    try:
        # TODO
        pass
    except TypeError:
        raise TypeError('A numpy array expected for x.')

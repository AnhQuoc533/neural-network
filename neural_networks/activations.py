import numpy as np


def sigmoid(x, derivative=False):
    try:
        x = 1 / (1 + np.exp(-x))
        if derivative:
            return x * (1 - x)
        return x

    except TypeError:
        raise TypeError('A numpy array expected for x.')


def tanh(x, derivative=False):
    try:
        if derivative:
            return 1 - np.tanh(x)**2
        return np.tanh(x)

    except TypeError:
        raise TypeError('A numpy array expected for x.')


def relu(x, derivative=False):
    try:
        if derivative:
            return np.float64(x > 0)  # Faster than np.where(x > 0, 1., 0.)
        return np.where(x < 0, 0., x)

    except TypeError:
        raise TypeError('A numpy array expected for x.')


def leaky_relu(x, derivative=False):
    try:
        if derivative:
            return np.where(x < 0, 0.01, 1.)
        return np.where(x < 0, 0.01*x, x)

    except TypeError:
        raise TypeError('A numpy array expected for x.')


def softmax(x, derivative=False):
    try:
        if derivative:
            # TODO
            ...
        exp_scores = np.exp(x)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    except TypeError:
        raise TypeError('A numpy array expected for x.')

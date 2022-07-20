import numpy as np


def log_loss(y_pred, y_true):
    if y_pred.shape != y_true.shape:
        raise ValueError('y_pred and y_true must have the same size.')
    if y_pred.ndim != 2:
        raise ValueError('2D arrays expected for y_pred and y_true.')

    loss = - y_true*np.log(y_pred) - (1-y_true)*np.log(1-y_pred)
    np.nan_to_num(loss, copy=False)
    return loss


def d_log_loss(y_pred, y_true):
    if y_pred.shape != y_true.shape:
        raise ValueError('y_pred and y_true must have the same size.')
    if y_pred.ndim != 2:
        raise ValueError('2D arrays expected for y_pred and y_true.')

    d_loss = - y_true/y_pred + (1-y_true)/(1-y_pred)
    np.nan_to_num(d_loss, copy=False)
    return d_loss


def cross_entropy_loss(y_pred, y_true):
    if y_pred.shape != y_true.shape:
        raise ValueError('y_pred and y_true must have the same size.')
    if y_pred.ndim != 2:
        raise ValueError('2D arrays expected for y_pred and y_true.')

    loss = - (y_true * np.log(y_pred)).sum(axis=1, keepdims=True)
    np.nan_to_num(loss, copy=False)
    return loss


def d_cross_entropy_loss(y_pred, y_true):
    if y_pred.shape != y_true.shape:
        raise ValueError('y_pred and y_true must have the same size.')
    if y_pred.ndim != 2:
        raise ValueError('2D arrays expected for y_pred and y_true.')

    # TODO


def quadratic_loss(y_pred, y_true):
    if y_pred.shape != y_true.shape:
        raise ValueError('y_pred and y_true must have the same size.')
    if y_pred.ndim != 2:
        raise ValueError('2D arrays expected for y_pred and y_true.')

    return (y_pred - y_true)**2 / 2


def d_quadratic_loss(y_pred, y_true):
    if y_pred.shape != y_true.shape:
        raise ValueError('y_pred and y_true must have the same size.')
    if y_pred.ndim != 2:
        raise ValueError('2D arrays expected for y_pred and y_true.')

    return y_pred - y_true

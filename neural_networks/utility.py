import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, y_min = X.min(axis=0) - 1
    x_max, y_max = X.max(axis=0) + 1
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.get_cmap('Spectral'))
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.get_cmap('Spectral'))

    plt.show()

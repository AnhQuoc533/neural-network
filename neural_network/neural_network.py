import matplotlib.pyplot as plt
from time import perf_counter
from .loss import *
from .activation import relu, sigmoid, tanh, softmax


class NeuralNetwork:
    __epsilon = 1e-8  # Adam

    def __init__(self, neurons: list[int], is_clf=True, seed: float = None):

        if not all(type(neuron) is int and neuron > 0 for neuron in neurons):
            raise ValueError('A list of positive integer expected for neurons.')

        self.__neurons = neurons[:]
        self.seed = np.random.RandomState(seed)
        self.is_clf = is_clf

        self.__parameters = {}
        self.__activations = []
        self.__loss = None
        self.__labels = None  # The true classes
        self.__costs = []

        self.__beta1 = 0.9  # Momentum
        self.__beta2 = 0.999  # RMSprop

    @property
    def neurons(self):
        return self.__neurons

    @property
    def size(self):
        return len(self.__neurons) + 1  # Number of layers from input to output

    @property
    def parameters(self):
        return self.__parameters

    @property
    def activations(self):
        return self.__activations

    @property
    def loss(self):
        return self.__loss

    @property
    def costs(self):
        return self.__costs

    @property
    def labels(self):
        return self.__labels

    def set_activations(self, activations: list = None):
        if len(self.__activations):
            self.__activations = []

        if activations is None:
            if self.is_clf:
                if self.neurons[-1] == 1:
                    self.__activations += [relu] * (len(self.__neurons) - 1)
                    self.__activations.append(sigmoid)
                else:
                    self.__activations += [tanh] * (len(self.__neurons) - 1)
                    self.__activations.append(softmax)

            else:
                # TODO: Activations for regression model
                ...

        # Check activations
        elif type(activations) is not list or not all(callable(a) for a in activations):
            raise TypeError('List of functions or callable objects expected for activations.')

        # Check activations length
        elif len(activations) != len(self.__neurons):
            raise ValueError(f'The length of activations must be {len(self.__neurons)}.')

        else:
            self.__activations.extend(activations)

    def set_params(self, n_att: int, initializer: str = None, weight: dict = None, bias: dict = None):
        if self.__costs:
            raise ValueError('Cannot change parameters after fitting the instance.')
        if n_att < 1:
            raise ValueError('A positive integer expected for n_att')
        layer_dims = [n_att] + self.neurons

        # Check and set weights
        if weight is None:
            for l in range(1, self.size):
                # TODO: Optimize code
                if initializer is None:
                    factor = 1
                elif initializer == 'he':  # For ReLU and variants
                    factor = np.sqrt(2 / layer_dims[l-1])
                elif initializer == 'lecun':  # For SELU
                    factor = np.sqrt(1 / layer_dims[l-1])
                elif initializer == 'xavier':  # For Tanh/Sigmoid/Softmax
                    factor = np.sqrt(2 / (layer_dims[l-1] + layer_dims[l]))
                else:
                    raise ValueError("initializer must be one of 'he', 'xavier', 'lecun'.")

                self.parameters[f'W{l}'] = self.seed.randn(layer_dims[l-1], layer_dims[l]) * factor

        elif type(weight) is not dict:
            raise TypeError('A dictionary expected for weight.')

        else:
            for l in range(1, self.size):
                try:
                    self.parameters[f'W{l}'] = np.array(weight[f'W{l}'], dtype=np.float64)
                    assert self.parameters[f'W{l}'].shape == (layer_dims[l-1], layer_dims[l])

                except KeyError:
                    raise ValueError(f'Missing weights for layer {l}.')
                except AssertionError:
                    raise ValueError(f'Array of weights for layer {l} must have the size of '
                                     f'{(layer_dims[l-1], layer_dims[l])}.')

        # Check and set biases
        if bias is None:
            for l in range(1, self.size):
                self.parameters[f'b{l}'] = np.zeros((1, layer_dims[l]), dtype=np.float64)

        elif type(bias) is not dict:
            raise TypeError('A dictionary expected for bias.')

        else:
            for l in range(1, self.size):
                dims = layer_dims[l]
                try:
                    self.parameters[f'b{l}'] = np.array(bias[f'b{l}'], dtype=np.float64)
                    if self.parameters[f'b{l}'].shape in {(dims,), (dims, 1)}:
                        self.parameters[f'b{l}'].reshape((1, -1))

                    assert self.parameters[f'b{l}'].shape == (1, dims)

                except KeyError:
                    raise ValueError(f'Missing biases for layer {l}.')
                except AssertionError:
                    raise ValueError(f'Array of bias for layer {l} must have the size of {(1, dims)} or {(dims,)}.')

    def set_loss(self, loss=None):
        if loss is None:
            if self.is_clf:
                self.__loss = log_loss if self.neurons[-1] == 1 else cross_entropy_loss
            else:
                self.__loss = quadratic_loss

        elif callable(loss):
            self.__loss = loss

        else:
            raise TypeError('function and function_derivative must be callable functions or objects.')

    @staticmethod
    def one_hot_encode(y, labels):
        return np.float64(np.asarray(y).reshape(-1, 1) == np.asarray(labels).ravel())

    def __validate_Xy(self, X, y):
        # Validate X
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError('2D array expected for the training data X.')
        if self.parameters and X.shape[1] != len(self.parameters['W1']):
            raise ValueError('The number of attributes in X does not match '
                             'the number of initialize weights in the first layer.')

        # Validate y
        y = np.asarray(y, dtype=np.float64)
        if len(y) != len(X):
            raise ValueError('X and y do not have the same number of samples.')
        elif y.ndim == 1:
            y = y.reshape((-1, 1))
        elif y.ndim != 2:
            raise ValueError('1D or 2D array expected for the label data y.')
        elif y.shape[1] != 1:
            raise ValueError('A column vector expected for the label data y. '
                             'Consider reshape, e.g. y.reshape((-1, 1)).')

        # Encode for multiclass prediction
        if self.is_clf and self.neurons[-1] != 1:
            if self.labels is None:
                y = self.one_hot_encode(y, range(self.neurons[-1]))
            elif len(self.labels) != self.neurons[-1]:
                raise ValueError('The length of labels does not match the output size.')
            else:
                y = self.one_hot_encode(y, self.labels)

        return X, y

    def __init_optimizer(self, optimizer: str = None):
        if optimizer is None:
            def f(alpha, grad, t):
                self.__basic_update(alpha, grad)

        elif optimizer == 'momentum':
            v = {'d'+key: np.zeros(value.shape) for key, value in self.__parameters.items()}
            def f(alpha, grad, t):
                self.__momentum_update(alpha, grad, v)

        elif optimizer == 'rmsprop':
            s = {'d'+key: np.zeros(value.shape) for key, value in self.__parameters.items()}
            def f(alpha, grad, t):
                self.__rmsprop_update(alpha, grad, s)

        elif optimizer == 'adam':
            v = {'d'+key: np.zeros(value.shape) for key, value in self.__parameters.items()}
            s = {key: np.zeros(value.shape) for key, value in v.items()}
            def f(alpha, grad, t):
                self.__adam_update(alpha, grad, v, s, t)

        else:
            raise ValueError("optimizer must be one of 'momentum', 'adam', 'rmsprop', 'adagrad' or None.")

        return f

    def _forward_pass(self, X, keep_prob: list = None):
        cache = []
        A_prev = X
        D = None

        for l in range(1, self.size):
            # Inverted dropout regularization
            if keep_prob is not None and l != 1:
                p = keep_prob[l-2]
                D = np.where(self.seed.rand(*A_prev.shape) < p, 1/p, .0)
                A_prev *= D  # Scale the value of kept neurons

            Z = A_prev @ self.parameters[f'W{l}'] + self.parameters[f'b{l}']
            cache.append((A_prev, Z) if D is None else (A_prev, Z, D))
            A_prev = self.activations[l-1](Z)

        return A_prev, cache

    def __compute_cost(self, y_pred, y_true, lambd=.0):
        cost = self.loss(y_pred, y_true).mean()

        # L2 regularization
        if lambd:
            reg = .0
            for l in range(1, self.size):
                reg += (self.parameters[f'W{l}']**2).sum()
            cost += lambd / (2 * len(y_true)) * reg

        return cost

    def __backward_pass(self, dA, cache, gradient: dict, lambd=.0):
        m = len(dA)
        for l in range(len(self.neurons), 0, -1):
            A_prev, Z, *D = cache[l-1]

            # Derivative of activation function
            dZ = self.activations[l-1](Z, derivative=True) * dA

            # Derivative of A_prev
            if l != 1:
                dA = dZ @ self.parameters[f'W{l}'].T
                # Inverted dropout regularization
                if D:
                    dA *= D[0]

            if lambd:
                gradient[f'dW{l}'] = (A_prev.T @ dZ + lambd * self.parameters[f'W{l}']) / m
            else:
                gradient[f'dW{l}'] = (A_prev.T @ dZ) / m
            gradient[f'db{l}'] = dZ.mean(axis=0, keepdims=True)

    def __momentum_update(self, lr: float, gradient: dict, v: dict):
        for l in range(1, self.size):
            v[f'dW{l}'] = self.__beta1 * v[f'dW{l}'] + (1 - self.__beta1) * gradient[f'dW{l}']
            v[f'db{l}'] = self.__beta1 * v[f'db{l}'] + (1 - self.__beta1) * gradient[f'db{l}']

            self.__parameters[f'W{l}'] -= lr * v[f'dW{l}']
            self.__parameters[f'b{l}'] -= lr * v[f'db{l}']

    def __rmsprop_update(self, lr: float, gradient: dict, s: dict):
        for l in range(1, self.size):
            s[f'dW{l}'] = self.__beta2 * s[f'dW{l}'] + (1 - self.__beta2) * gradient[f'dW{l}'] ** 2
            s[f'db{l}'] = self.__beta2 * s[f'db{l}'] + (1 - self.__beta2) * gradient[f'db{l}'] ** 2

            self.__parameters[f'W{l}'] -= lr * gradient[f'dW{l}'] / (np.sqrt(s[f'dW{l}']) + self.__epsilon)
            self.__parameters[f'b{l}'] -= lr * gradient[f'db{l}'] / (np.sqrt(s[f'db{l}']) + self.__epsilon)

    def __adam_update(self, lr: float, gradient: dict, v: dict, s: dict, t: int):
        v_correct = 1 - self.__beta1 ** t
        s_correct = 1 - self.__beta2 ** t
            
        for l in range(1, self.size):
            v[f'dW{l}'] = self.__beta1 * v[f'dW{l}'] + (1 - self.__beta1) * gradient[f'dW{l}']
            v[f'db{l}'] = self.__beta1 * v[f'db{l}'] + (1 - self.__beta1) * gradient[f'db{l}']

            s[f'dW{l}'] = self.__beta2 * s[f'dW{l}'] + (1 - self.__beta2) * gradient[f'dW{l}']**2
            s[f'db{l}'] = self.__beta2 * s[f'db{l}'] + (1 - self.__beta2) * gradient[f'db{l}']**2

            self.__parameters[f'W{l}'] -= lr * (v[f'dW{l}']/v_correct) / (np.sqrt(s[f'dW{l}']/s_correct) + self.__epsilon)
            self.__parameters[f'b{l}'] -= lr * (v[f'db{l}']/v_correct) / (np.sqrt(s[f'db{l}']/s_correct) + self.__epsilon)

    def __basic_update(self, lr: float, gradient: dict):
        for l in range(len(self.neurons), 0, -1):
            self.__parameters[f'W{l}'] -= lr * gradient[f'dW{l}']
            self.__parameters[f'b{l}'] -= lr * gradient[f'db{l}']

    def fit(
        self,
        X,
        y,
        learning_rate: float,
        epochs: int,
        verbose=False,
        plot_cost=False,
        step=100,
        labels: list = None,
        lambd=.0,
        keep_prob=None,
        batch_size: int = None,
        shuffle=False,
        optimizer: str = None,
        decay_rate=0
    ):

        # Check X, y
        self.__labels = labels
        X, y = self.__validate_Xy(X, y)
        m = len(X)

        # Check inverted dropout parameter
        if keep_prob is not None:
            if type(keep_prob) not in [float, list]:
                raise TypeError('A float number or a list of float numbers expected for keep_prob.')
            elif type(keep_prob) is float:
                if not (0 < keep_prob <= 1):
                    raise ValueError('keep_prob must be in the range 0 (exclusive) - 1 (inclusive).')
                else:
                    keep_prob = [keep_prob] * (len(self.neurons) - 1)
            elif len(keep_prob) != len(self.neurons)-1:
                raise ValueError(f'The length of keep_prob must be {len(self.neurons)-1}.')
            elif not all(type(p) is float and 0 <= p <= 1 for p in keep_prob):
                raise ValueError(f'Each element in keep_prob must be a float number and in the range 0 - 1.')

        # Check batch_size
        if batch_size is None:
            batch_size = len(X)
        elif batch_size > len(X):
            raise ValueError('batch_size cannot exceed the number of samples.')
        elif batch_size < 1:
            raise ValueError('A positive integer expected for batch_size.')
        n_batches = (m // batch_size + 1) if m % batch_size else (m // batch_size)

        # Initialize weights and biases
        if not self.parameters:
            self.set_params(X.shape[1], 'he' if self.neurons[-1] == 1 else 'xavier')

        # Initialize activation functions
        if not self.activations:
            self.set_activations()

        # Initialize loss function
        if self.loss is None:
            self.set_loss()

        # Initialize optimizer
        update_parameters = self.__init_optimizer(optimizer)

        # Initialize supporting variables
        self.__costs = []
        gradient = {}
        t = 0

        start_time = perf_counter()
        for i in range(epochs):
            cost = .0
            if shuffle:
                new_id = self.seed.permutation(m)
                X = X[new_id]
                y = y[new_id]

            for b in range(n_batches):
                # Partition
                mini_X = X[b*batch_size : (b+1)*batch_size]
                mini_y = y[b*batch_size : (b+1)*batch_size]

                # Forward propagation
                y_pred, cache = self._forward_pass(mini_X, keep_prob)

                # Compute mini-cost
                if i % step == 0:
                    cost += self.__compute_cost(y_pred, mini_y, lambd)

                # Backward propagation
                # dA = self.loss(y_pred, mini_y, derivative=True)
                self.__backward_pass(self.loss(y_pred, mini_y, derivative=True), cache, gradient, lambd)

                # Update parameters with optimizer
                update_parameters(learning_rate, gradient, t := t+1)

            # Compute and print cost
            if i % step == 0:
                cost /= n_batches
                self.__costs.append(cost)
                if verbose:
                    print(f"Cost after epoch {i:{len(str(epochs))}}: {cost}")
        end_time = perf_counter()

        if verbose:
            print("\nTime elapsed:", end_time - start_time, 'seconds.')

        if plot_cost:
            plt.plot(self.__costs)
            plt.ylabel('cost')
            plt.xlabel(f'epochs (per {step})')
            plt.show()

        return self

    def predict(self, X):
        # Check X
        if not self.costs:
            raise ValueError('Fit the training data to the instance first.')

        A_prev = np.asarray(X, dtype=np.float64)
        if A_prev.ndim != 2:
            raise ValueError('2D array expected for X.')
        if A_prev.shape[1] != len(self.parameters['W1']):
            raise ValueError('The number of attributes in X does not match '
                             'the number of weights in the first layer of the neural network.')

        # Forward pass
        for l in range(1, self.size):
            Z = A_prev @ self.parameters[f'W{l}'] + self.parameters[f'b{l}']
            A_prev = self.activations[l-1](Z)

        # Prediction
        if self.is_clf:
            if A_prev.shape[1] == 1:
                return np.int64(A_prev > 0.5).ravel()
            elif self.__labels is None:
                return A_prev.argmax(axis=1)
            else:
                y_pred = A_prev.argmax(axis=1)
                return np.select([y_pred == i for i in range(len(self.__labels))], self.__labels, -1)

        else:
            return A_prev

    def accuracy_score(self, X, y):
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError('1D array expected for y.')
        if len(y) != len(X):
            raise ValueError('X and y do not have the same length.')

        return np.count_nonzero(self.predict(X) == y) / y.size

    def check_gradient(self, X, y, lambd=.0, epsilon=1e-7):
        A_prev, y = self.__validate_Xy(X, y)
        backup = {'params': {key: value.copy() for key, value in self.parameters.items()}}

        # Initialize weights and biases
        if not self.parameters:
            self.set_params(A_prev.shape[1], 'he' if self.neurons[-1] == 1 else 'xavier')

        # Initialize activation functions
        if not self.__activations:
            backup['a'] = True
            self.set_activations()

        # Initialize loss function
        if self.__loss is None:
            backup['loss'] = True
            self.set_loss()

        # Forward propagation
        y_pred, cache = self._forward_pass(X)

        # Compute approximation of gradients
        grad_approx = []
        for param in self.parameters.values():
            for i in range(param.size):
                tmp_param = param.flat[i]

                # Left side
                param.flat[i] = tmp_param + epsilon
                cost_plus = self.__compute_cost(self._forward_pass(X)[0], y, lambd)

                # Right side
                param.flat[i] = tmp_param - epsilon
                cost_minus = self.__compute_cost(self._forward_pass(X)[0], y, lambd)

                grad_approx.append((cost_plus - cost_minus) / (2 * epsilon))
                param.flat[i] = tmp_param

        # Backward propagation
        gradients = {}
        self.__backward_pass(self.__loss(y_pred, y, derivative=True), cache, gradients, lambd)
        grad = tuple(gradients['d'+key].ravel() for key in self.__parameters)
        grad = np.concatenate(grad)

        # Reset
        self.__parameters = backup['params']
        if 'a' in backup:
            self.__activations = []
        if 'loss' in backup:
            self.__loss = None

        assert np.allclose(grad, grad_approx), 'There is an error in the backward propagation! ' \
                                               'Please check the derivative of activation/loss functions.'
        print('No error found.')

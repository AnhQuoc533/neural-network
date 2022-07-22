import pytest
import h5py
from sklearn.datasets import make_circles, make_moons
from ..neural_network import NeuralNetwork
from ..activations import *

# Load 2D dataset
data = np.load('datasets/data.npz', allow_pickle=True)

# Load cat images dataset
with h5py.File('datasets/train_catvnoncat.h5', "r") as train_dataset:
    train_set_x = np.array(train_dataset["train_set_x"])
    train_set_y = np.array(train_dataset["train_set_y"]).T

with h5py.File('datasets/test_catvnoncat.h5', "r") as test_dataset:
    test_set_x = np.array(test_dataset["test_set_x"])
    test_set_y = np.array(test_dataset["test_set_y"])
    classes = np.array(test_dataset["list_classes"]).T

# Flatten
train_set_x = train_set_x.reshape(train_set_x.shape[0], -1)
test_set_x = test_set_x.reshape(test_set_x.shape[0], -1)

# Normalize
train_set_x = train_set_x / 255
test_set_x = test_set_x / 255


def load_planar_dataset():
    np.random.seed(1)
    m = 400  # number of examples
    N = int(m / 2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    Y = np.zeros(m, dtype=np.uint8)  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    return X, Y


def load_2d_dataset():
    return data['X'], data['y'], data['Xval'], data['yval']


def test_logistic_regression():
    model = NeuralNetwork(neurons=[1])
    model.set_params(train_set_x.shape[1], weight={'W1': np.zeros((train_set_x.shape[1], 1))})

    model.fit(train_set_x, train_set_y, learning_rate=0.005, epochs=2000, step=100)

    # Check the costs
    assert model.costs[0] == 0.6931471805599453
    assert model.costs[-1] == 0.14087207570310165

    # Check accuracy score
    assert model.accuracy_score(train_set_x, train_set_y) == 0.9904306220095693
    assert model.accuracy_score(test_set_x, test_set_y) == 0.70


def test_shallow_nn_1():
    X, Y = load_planar_dataset()
    np.random.seed(2)
    neurons = 4

    model = NeuralNetwork([neurons, 1])

    weight = {'W1': np.random.randn(neurons, X.shape[1]).T * 0.01,
              'W2': np.random.randn(neurons, 1) * 0.01}
    model.set_params(X.shape[1], weight=weight)
    model.set_activations([tanh, sigmoid])

    model.fit(X, Y, learning_rate=1.2, epochs=10_000, step=1_000)

    # Check the costs
    assert pytest.approx(model.costs[0]) == 0.6930480201239823
    assert pytest.approx(model.costs[1]) == 0.28808329356901846
    assert pytest.approx(model.costs[2]) == 0.25438549407324607

    assert pytest.approx(model.costs[-1]) == 0.21853822797177622
    assert pytest.approx(model.costs[-2]) == 0.2193839544384549

    # Check accuracy score
    assert model.accuracy_score(X, Y) == 0.9075


def test_shallow_nn_2():
    model = NeuralNetwork([7, 1])
    np.random.seed(1)
    weight = {'W1': np.random.randn(7, train_set_x.shape[1]).T * 0.01,
              'W2': np.random.randn(7, 1) * 0.01}
    model.set_params(train_set_x.shape[1], weight=weight)

    model.fit(train_set_x, train_set_y, learning_rate=0.0075, epochs=2500, step=100)

    # Check the costs
    assert model.costs[0] == 0.6930497356599891
    assert model.costs[1] == 0.6464320953428849
    assert model.costs[2] == 0.6325140647912677

    assert pytest.approx(model.costs[-3]) == 0.05919329501038164
    assert pytest.approx(model.costs[-2]) == 0.05336140348560552
    assert pytest.approx(model.costs[-1]) == 0.04855478562877014

    # Check accuracy score
    assert model.accuracy_score(train_set_x, train_set_y) == 1
    assert model.accuracy_score(test_set_x, test_set_y) == 0.72


def initialize_parameters(layer_dims, seeds=1):
    np.random.seed(seeds)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]).T / np.sqrt(layer_dims[l-1])

    return parameters


def test_deep_nn_1():
    model = NeuralNetwork([20, 7, 5, 1])
    weight = initialize_parameters([train_set_x.shape[1], 20, 7, 5, 1])
    model.set_params(train_set_x.shape[1], weight=weight)

    model.fit(train_set_x, train_set_y, learning_rate=0.0075, epochs=2500, step=100)

    # Check the costs
    assert pytest.approx(model.costs[0]) == 0.7717493284237686
    assert pytest.approx(model.costs[1]) == 0.6720534400822913
    assert pytest.approx(model.costs[2]) == 0.6482632048575212

    assert pytest.approx(model.costs[-3]) == 0.10285466069352679
    assert pytest.approx(model.costs[-2]) == 0.10089745445261786
    assert pytest.approx(model.costs[-1]) == 0.09287821526472397

    # Check accuracy score
    assert pytest.approx(model.accuracy_score(train_set_x, train_set_y)) == 0.9856459330143541
    assert model.accuracy_score(test_set_x, test_set_y) == 0.8


def test_deep_nn_2():
    train_x, train_y, test_x, test_y = load_2d_dataset()

    model = NeuralNetwork([20, 3, 1])
    weight = initialize_parameters([train_x.shape[1], 20, 3, 1], 3)
    model.set_params(train_x.shape[1], weight=weight)

    model.fit(train_x, train_y, learning_rate=0.3, epochs=30_000, step=10_000)

    # plt.rcParams['figure.figsize'] = (7.0, 4.0)
    # axes = plt.gca()
    # axes.set_xlim([-0.75, 0.40])
    # axes.set_ylim([-0.75, 0.65])

    # plt.title("Model without regularization")
    # plot_decision_boundary(model.predict, train_x, train_y)

    # Check the costs
    assert pytest.approx(model.costs) == [0.6557412523481002, 0.1632998752572421, 0.13851642423244123]

    # Check accuracy score
    assert pytest.approx(model.accuracy_score(train_x, train_y)) == 0.9478672985781991
    assert model.accuracy_score(test_x, test_y) == 0.915


def test_L2_regularization():
    train_x, train_y, test_x, test_y = load_2d_dataset()

    model = NeuralNetwork([20, 3, 1])
    weight = initialize_parameters([train_x.shape[1], 20, 3, 1], 3)
    model.set_params(train_x.shape[1], weight=weight)

    model.fit(train_x, train_y, learning_rate=0.3, lambd=0.7, epochs=30_000, step=10_000)

    # plt.rcParams['figure.figsize'] = (7.0, 4.0)
    # axes = plt.gca()
    # axes.set_xlim([-0.75, 0.40])
    # axes.set_ylim([-0.75, 0.65])

    # plt.title("Model with L2 regularization")
    # plot_decision_boundary(model.predict, train_x, train_y)

    # Check the costs
    assert pytest.approx(model.costs) == [0.6974484493131264, 0.2684918873282238, 0.2680916337127302]

    # Check accuracy score
    assert model.accuracy_score(train_x, train_y) == 0.9383886255924171
    assert model.accuracy_score(test_x, test_y) == 0.93


def test_forward_pass_dropout():
    np.random.seed(1)
    X_assess = np.random.randn(3, 5).T

    W1 = np.random.randn(2, 3).T
    b1 = np.random.randn(2, 1).T

    W2 = np.random.randn(3, 2).T
    b2 = np.random.randn(3, 1).T

    W3 = np.random.randn(1, 3).T
    b3 = np.random.randn(1, 1)

    model = NeuralNetwork([2, 3, 1], seed=1)
    model.set_params(X_assess.shape[1], weight={"W1": W1, "W2": W2, "W3": W3}, bias={"b1": b1, "b2": b2, "b3": b3})
    model.set_activations()

    # Check forward propagation
    y, _ = model.forward_pass(X_assess, [.7]*2)
    assert np.allclose(y, [[0.36974721], [0.49683389], [0.04565099], [0.01446893], [0.36974721]])


def test_dropout():
    train_x, train_y, test_x, test_y = load_2d_dataset()

    model = NeuralNetwork([20, 3, 1], seed=24)
    weight = initialize_parameters([train_x.shape[1], 20, 3, 1], 3)
    model.set_params(train_x.shape[1], weight=weight)

    model.fit(train_x, train_y, learning_rate=0.3, keep_prob=0.8, epochs=30_000, step=10_000)

    # plt.rcParams['figure.figsize'] = (7.0, 4.0)
    # axes = plt.gca()
    # axes.set_xlim([-0.75, 0.40])
    # axes.set_ylim([-0.75, 0.65])

    # plt.title("Model with dropout")
    # plot_decision_boundary(model.predict, train_x, train_y)

    # Check the costs
    assert pytest.approx(model.costs) == [0.6585998432035588, 0.20187690743859132, 0.15421839870569154]

    # Check accuracy score
    assert pytest.approx(model.accuracy_score(train_x, train_y)) == 0.957345971563981
    assert model.accuracy_score(test_x, test_y) == 0.94


def test_zero_params():
    train_x, train_y = make_circles(n_samples=300, noise=.05, random_state=1)
    test_X, test_Y = make_circles(n_samples=100, noise=.05, random_state=2)

    model = NeuralNetwork([20, 3, 1])
    weights = {}
    layer_dims = [train_x.shape[1], 20, 3, 1]

    for l in range(1, 4):
        weights['W' + str(l)] = np.zeros((layer_dims[l - 1], layer_dims[l]))
    model.set_params(train_x.shape[1], weight=weights)

    model.fit(train_x, train_y, learning_rate=0.01, epochs=15_000, step=1_000)

    assert all(pytest.approx(x) == 0.6931471805599453 for x in model.costs)
    assert model.accuracy_score(train_x, train_y) == 0.5
    assert model.accuracy_score(test_X, test_Y) == 0.5


def test_big_weights():
    train_x, train_y = make_circles(n_samples=300, noise=.05, random_state=1)
    test_X, test_Y = make_circles(n_samples=100, noise=.05, random_state=2)

    model = NeuralNetwork([10, 5, 1])
    weights = {}
    layer_dims = [train_x.shape[1], 10, 5, 1]

    np.random.seed(3)
    for l in range(1, 4):
        weights['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]).T * 10
    model.set_params(train_x.shape[1], weight=weights)

    model.fit(train_x, train_y, learning_rate=0.01, epochs=15_000, step=1_000)

    # Check the costs
    assert model.costs[0] == float("inf")

    # Check accuracy score
    assert model.accuracy_score(train_x, train_y) == 0.5 == model.accuracy_score(test_X, test_Y)


def test_he_init():
    train_x, train_y = make_circles(n_samples=300, noise=.05, random_state=1)
    test_X, test_Y = make_circles(n_samples=100, noise=.05, random_state=2)

    model = NeuralNetwork([10, 5, 1], seed=24)
    model.fit(train_x, train_y, learning_rate=0.01, epochs=15_000, step=1_000)

    # Check the costs
    assert pytest.approx(model.costs[0]) == 0.712495051645707
    assert pytest.approx(model.costs[1]) == 0.6732084171360789
    assert pytest.approx(model.costs[2]) == 0.6491058169126

    assert pytest.approx(model.costs[-3]) == 0.08220191573225748
    assert pytest.approx(model.costs[-2]) == 0.07588034042714038
    assert pytest.approx(model.costs[-1]) == 0.07088313671881122

    # Check accuracy score
    assert model.accuracy_score(train_x, train_y) == 0.98
    assert model.accuracy_score(test_X, test_Y) == 0.96


def test_gradient_check_1():
    train_x, train_y, _, _ = load_2d_dataset()
    model = NeuralNetwork([5, 3, 1], seed=10)

    # Without L2 regularization
    model.check_gradient(train_x, train_y)
    assert model.costs == []
    assert len(model.parameters) == 0
    assert len(model.activations) == 0

    # With L2 regularization
    model.check_gradient(train_x, train_y, lambd=0.02)
    assert model.costs == []
    assert len(model.parameters) == 0
    assert len(model.activations) == 0


def test_gradient_check_2():
    w = {}
    b = {}
    np.random.seed(1)

    x = np.random.randn(100, 4)
    y = np.random.randint(2, size=100)

    w['W1'] = np.random.randn(4, 5)
    w['W2'] = np.random.randn(5, 3)
    w['W3'] = np.random.randn(3, 1)
    b['b2'] = np.random.randn(1, 3)
    b['b1'] = np.random.randn(1, 5)
    b['b3'] = np.random.randn(1, 1)

    backup_w = {key: value.copy() for key, value in w.items()}
    backup_b = {key: value.copy() for key, value in b.items()}

    model = NeuralNetwork([5, 3, 1])
    model.set_params(n_att=4, weight=w, bias=b)

    # Without L2 regularization
    model.check_gradient(x, y)
    assert model.costs == []
    assert len(model.activations) == 0
    assert all(np.all(value == model.parameters[key]) for key, value in backup_w.items())
    assert all(np.all(value == model.parameters[key]) for key, value in backup_b.items())

    # With L2 regularization
    model.check_gradient(x, y, lambd=0.05)
    assert model.costs == []
    assert len(model.activations) == 0
    assert all(np.all(value == model.parameters[key]) for key, value in backup_w.items())
    assert all(np.all(value == model.parameters[key]) for key, value in backup_b.items())


def test_gradient_check_3():
    x = np.random.randn(100, 4)
    y = np.random.randint(2, size=100)
    model = NeuralNetwork([5, 3, 1], seed=24)

    # Wrong derivative
    def false_relu(z, derivative=False):
        if derivative:
            return np.where(z < 0, 0.01, 1.)
        return np.where(z < 0, 0., z)

    activations = [tanh, false_relu, sigmoid]
    model.set_activations(activations)

    with pytest.raises(AssertionError):
        model.check_gradient(x, y)
    assert model.costs == []
    assert len(model.parameters) == 0
    assert len(model.activations) == 3
    assert model.activations == activations

    # Correct derivative
    activations = [tanh, relu, sigmoid]
    model.set_activations(activations)
    model.check_gradient(x, y)

    assert model.costs == []
    assert len(model.parameters) == 0
    assert len(model.activations) == 3
    assert model.activations == activations


def test_mini_batch():
    train_X, train_Y = make_moons(n_samples=300, noise=.2, random_state=3)

    model = NeuralNetwork([5, 2, 1], seed=3)
    model.fit(train_X, train_Y, learning_rate=0.0007, epochs=5000, step=1000, batch_size=64, shuffle=True)

    assert pytest.approx(model.costs) == [0.7588353088246047, 0.5977185220351452, 0.5316795298099085,
                                          0.49401085257243593, 0.4635011585260269]
    assert pytest.approx(model.accuracy_score(train_X, train_Y)) == 0.8066666666666666

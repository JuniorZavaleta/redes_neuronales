import matplotlib.pyplot as plt
import numpy as np

from Perceptron import step_function, Perceptron


def print_plot(perceptron: Perceptron):
    plt.scatter(perceptron.values_x[:, 0],
                perceptron.values_x[:, 1])
    w1, w2 = perceptron.weights[0], perceptron.weights[1]
    # x1 = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    x1 = np.array(range(10))
    x2 = (perceptron.threshold - (x1 * w1)) / w2
    plt.plot(x2)


X = np.array([
    [1.5, -0.3],
    [0.9, 0.05],
    [2.1, 0.2],
    [0.24, -0.87],
    [0.45, -0.60],
    [0.15, -0.43]
])
Y = np.array(
    [1,
     1,
     1,
     0,
     0,
     0])
W = [0, 0]
fruit_perceptron = Perceptron(values_x=X,
                              values_y=Y,
                              weights=W,
                              bias=0,
                              threshold=0,
                              learning_rate=0.1,
                              activation_function=step_function,
                              message="Pesos finales ->",
                              debug=True)
fruit_perceptron.training()
print_plot(fruit_perceptron)
plt.show()

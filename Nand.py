import matplotlib.pyplot as plt
import numpy as np

from Perceptron import print_line, step_function, Perceptron

# NAND
bias = 0
X = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

print("NAND")
print_line()
Y = [1, 1, 1, 0]

perceptron_nand = Perceptron(values_x=X,
                             values_y=Y,
                             weights=[0, 1, 0.3],
                             bias=0,
                             threshold=0.5,
                             activation_function=step_function,
                             learning_rate=0.1,
                             message="Final Weights NAND for 2 elements")
perceptron_nand.training()
# perceptron_nand.weights
plt.scatter(X[:, 1],
            X[:, 2],
            c=Y)
w0, w1, w2 = perceptron_nand.weights[0], perceptron_nand.weights[1], perceptron_nand.weights[2]

x0 = np.array([1, 1, 1])
x1 = np.array([-1, 0, 1])

x2 = ((x0 * w0) - (x1 * w1)) / w2
plt.plot(x2)
plt.show()

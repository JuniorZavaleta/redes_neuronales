# Port Or and And
import numpy as np


def step_function(y, threshold):
    return 0 if y < threshold else 1


def print_line():
    print("-" * 50)


class Perceptron:
    W = []

    def __init__(self, X, Y, W, threshold, activation_function, learning_rate, bias,
                 message=None, debug=True):
        """
        :param X: Input values
        :param Y: Expected values
        :param W: Weight for each value
        :param threshold:
        :param activation_function: Function to activate :v
        :param message Title when the perceptron show the results of training
        """
        self.X = X
        self.Y = Y
        self.W = W
        self.threshold = threshold
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.bias = bias
        self.message = message if message is not None else "Results"
        self.debug = debug

    def training(self):
        has_error = True
        it = 1
        while has_error:
            has_error = False

            if self.debug:
                print("# Iteration " + str(it))
                print("X    |d |y |W")

            for i in range(len(self.X)):
                X = self.X[i]
                y_arr = []
                Y_arr = []
                W_arr = []
                for j in range(len(self.Y)):
                    Y = self.Y[j][i]
                    Y_arr.append(Y)
                    W = self.W[j]
                    W_arr.append(W)

                    dot_product = np.dot(X, W)
                    # dot_product = sum(value * weight for value, weight in zip(self.X[i], self.W))
                    y = self.activation_function(dot_product + self.bias[j], self.threshold)
                    error = Y - y
                    y_arr.append(y)

                    if error != 0:
                        self.bias[j] = self.bias[j] + (error * self.learning_rate)
                        for k in range(len(X)):
                            self.W[j][k] = W[k] + (error * X[k] * self.learning_rate)
                        has_error = True

                if self.debug:
                    desired_values = '[{0}]'.format(' '.join(str(e) for e in y_arr))
                    results_values = '[{0}]'.format(' '.join(str(e) for e in Y_arr))
                    weights = ' '.join(str(e) for e in W_arr)
                    print('{0}  {1}  {2}  {3}  {4}'.format(self.X[i],
                                                           desired_values,
                                                           results_values,
                                                           weights,
                                                           self.bias))
                    # print(self.X[i], desired_values, results_values, W_arr, self.bias)

            it = it + 1
        self.print_results_training()

    def evaluate(self, test_input):
        dot_product = sum(value * weight for value, weight in zip(test_input, self.W))
        y = self.activation_function(dot_product + self.bias)
        return y

    def print_results_training(self):
        print_line()
        print("{0}: {1}".format(self.message, self.W))
        print("Bias: {0}".format(self.bias))
        print_line()

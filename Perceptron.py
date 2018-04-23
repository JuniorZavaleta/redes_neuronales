# Port Or and And


def step_function(y, threshold):
    return 1 if y > threshold else 0


def print_line():
    print("-" * 50)


class Perceptron:
    weights = []

    def __init__(self, values_x, values_y, weights, threshold, activation_function, learning_rate, bias,
                 message=None):
        """
        :param values_x: Input values
        :param values_y: Expected values
        :param weights: Weight for each value
        :param threshold:
        :param activation_function: Function to activate :v
        :param message Title when the perceptron show the results of training
        """
        self.values_x = values_x
        self.values_y = values_y
        self.weights = weights
        self.threshold = threshold
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.bias = bias
        self.message = message if message is not None else "Results"

    def training(self):
        has_error = True
        it = 1
        while has_error:
            has_error = False
            print("# Iteration " + str(it))
            print("X    |d |y |W")

            for i in range(len(self.values_x)):
                dot_product = sum(value * weight for value, weight in zip(self.values_x[i], self.weights))
                y = self.activation_function(dot_product + self.bias, self.threshold)
                error = self.values_y[i] - y

                print("{0}|{1} |{2} |{3}".format(self.values_x[i], self.values_y[i], y, self.weights))

                if error != 0:
                    self.threshold = self.threshold - (error * self.values_y[i] * self.learning_rate)
                    for j in range(len(self.values_x[i])):
                        self.weights[j] = self.weights[j] + (error * self.values_x[i][j] * self.learning_rate)
                    has_error = True

            it = it + 1
        self.print_results_training()

    def print_results_training(self):
        print_line()
        print("{0}: {1}".format(self.message, self.weights))
        print('TH: {0}'.format(self.threshold))
        print_line()


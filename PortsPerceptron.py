# Port Or and And


def step_function(y):
    return 1 if y > 0 else 0


def print_line():
    print("-" * 50)


class Perceptron:
    weights = []

    def __init__(self, values_x, values_y, weights, threshold, activation_function):
        """
        :param values_x: Input values
        :param values_y: Expected values
        :param weights: Weight for each value
        :param threshold:
        :param activation_function: Function to activate :v
        """
        self.values_x = values_x
        self.values_y = values_y
        self.weights = weights
        self.threshold = threshold
        self.activation_function = activation_function

    def training(self):
        has_error = True
        it = 1
        while has_error:
            has_error = False
            print("# Iteration " + str(it))
            print("X{0}|d |y | W".format((len(self.weights)*3-1)*" "))
            for i in range(len(self.values_x)):
                dot_product = sum(value * weight for value, weight in zip(self.values_x[i], self.weights))
                dot_product = dot_product + (self.values_y[i] * self.threshold)
                y = self.activation_function(dot_product)
                error = self.values_y[i] - y

                print("{0}|{1} |{2} |{3}".format(self.values_x[i], self.values_y[i], y, self.weights))

                if error != 0:
                    self.threshold = self.threshold + (error * self.values_y[i])
                    for j in range(len(self.values_x[i])):
                        self.weights[j] = self.weights[j] + (error * self.values_x[i][j])
                    has_error = True

            it = it + 1
            # has_error = False


# Port with 2 elements
W = [0, -1]
b = 0.25

X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

# AND
print("AND")
print_line()
Y = [0, 0, 0, 1]
perceptron_and = Perceptron(X, Y, W, b, step_function)
perceptron_and.training()
print_line()
print("Final Weights OR for 3 elements: {0}".format(perceptron_and.weights))
print_line()

# OR
Y = [0, 1, 1, 1]
print("OR")
print_line()
perceptron_or = Perceptron(X, Y, W, b, step_function)
perceptron_or.training()
print_line()
print("Final Weights OR for 3 elements: {0}".format(perceptron_or.weights))
print_line()

# Port with 2 elements
X3 = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
]
W3 = [-1, -1, 0]
b3 = -0.25

# OR 3 elements
Y3 = [0, 1, 1, 1, 1, 1, 1, 1]
print("OR 3 elements")
print_line()
perceptron_or_3 = Perceptron(X3, Y3, W3, b3, step_function)
perceptron_or_3.training()
print_line()
print("Final Weights OR for 3 elements: {0}".format(perceptron_or_3.weights))
print_line()

# AND 3 elements
Y3 = [0, 0, 0, 0, 0, 0, 0, 1]
print("AND 3 elements")
print_line()
perceptron_and_3 = Perceptron(X3, Y3, W3, b3, step_function)
perceptron_and_3.training()
print_line()
print("Final Weights AND for 3 elements: {0}".format(perceptron_and_3.weights))
print_line()

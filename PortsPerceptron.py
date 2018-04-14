# Port Or and And


def step_function(y):
    return 1 if y > 0 else 0


def print_line():
    print("-" * 50)


class Perceptron:
    weights = []

    def __init__(self, values_x, values_y, weights, threshold, activation_function, message=None):
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
        self.message = message if message is not None else "Results"

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
        self.print_results_training()

    def print_results_training(self):
        print_line()
        print("{0}: {1}".format(self.message, self.weights))
        print_line()


# Port with 2 elements
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
perceptron_and = Perceptron(values_x=X,
                            values_y=Y,
                            weights=[0, -1],
                            threshold=0.25,
                            activation_function=step_function,
                            message="Final Weights AND for 2 elements")
perceptron_and.training()

# OR
Y = [0, 1, 1, 1]
print("OR")
print_line()
perceptron_or = Perceptron(values_x=X,
                           values_y=Y,
                           weights=[0, -1],
                           threshold=0.25,
                           activation_function=step_function,
                           message="Final Weights OR for 2 elements")
perceptron_or.training()

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

# OR 3 elements
Y3 = [0, 1, 1, 1, 1, 1, 1, 1]
print("OR 3 elements")
print_line()
perceptron_or_3 = Perceptron(values_x=X3,
                             values_y=Y3,
                             weights=[-1, -1, 0],
                             threshold=-0.25,
                             activation_function=step_function)
perceptron_or_3.training()

# AND 3 elements
Y3 = [0, 0, 0, 0, 0, 0, 0, 1]
print("AND 3 elements")
print_line()
perceptron_and_3 = Perceptron(values_x=X3,
                              values_y=Y3,
                              weights=[-1, -1, 0],
                              threshold=-0.25,
                              activation_function=step_function)
perceptron_and_3.training()

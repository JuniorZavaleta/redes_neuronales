# Port Or and And


def perceptron(weights, values_x, values_y, threshold):
    has_error = True
    it = 1
    while has_error:
        has_error = False
        print("#" + str(it))
        print("X1|X2|d |y")
        for i in range(len(X)):
            sum_x = values_x[i][0]*W[0] + X[i][1] * weights[1] + values_y[i] * threshold
            y = 1 if (sum_x > 0) else 0
            error = values_y[i] - y
            print("{} |{} |{} |{} ".format(
                values_x[i][0], values_x[i][1], values_y[i], y))
            if error != 0:
                threshold = threshold + (error * values_y[i])
                weights[0] = weights[0] + (error * X[i][0])
                weights[1] = weights[1] + (error * X[i][1])
                has_error = True

        it = it + 1


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
perceptron(W, X, [0, 0, 0, 1], b)
# OR
print("OR")
perceptron(W, X, [0, 1, 1, 1], b)

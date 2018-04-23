import matplotlib.pyplot as plt
import numpy as np

from Perceptron import step_function, Perceptron


def print_plot(perceptron: Perceptron):
    plt.scatter(perceptron.values_x[:, 0],
                perceptron.values_x[:, 1])
    w1, w2 = perceptron.weights[0], perceptron.weights[1]
    x1 = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    # x1 = np.array(range(10))
    x2 = (perceptron.threshold - (x1 * w1)) / w2
    plt.plot(x2)


# P1 = [0, 2]
P1_X = np.array([[0, 2]])
perceptron_p1 = Perceptron(values_x=P1_X,
                           values_y=[1],
                           weights=[2, 1],
                           bias=0,
                           threshold=0,
                           learning_rate=0.1,
                           activation_function=step_function,
                           message="P1 pesos finales ->",
                           debug=False)
perceptron_p1.training()
print_plot(perceptron_p1)

# P2 = [-2, 2]
P2_X = np.array([[-2, 2]])
perceptron_p2 = Perceptron(values_x=P2_X,
                           values_y=[1],
                           weights=[2, 1],
                           bias=0,
                           threshold=0,
                           learning_rate=0.1,
                           activation_function=step_function,
                           message="P2 pesos finales ->",
                           debug=False)
perceptron_p2.training()
print_plot(perceptron_p2)

# P3 = [-2, 2]
P3_X = np.array([[-2, 2]])
perceptron_p3 = Perceptron(values_x=P3_X,
                           values_y=[1],
                           weights=[2, 1],
                           bias=0,
                           threshold=0,
                           learning_rate=0.1,
                           activation_function=step_function,
                           message="P3 pesos finales ->",
                           debug=False)
perceptron_p3.training()
print_plot(perceptron_p3)

# P4 = [-2, -2]
P4_X = np.array([[-2, -2]])
perceptron_p4 = Perceptron(values_x=P4_X,
                           values_y=[1],
                           weights=[2, 1],
                           bias=0,
                           threshold=0,
                           learning_rate=0.1,
                           activation_function=step_function,
                           message="P4 pesos finales ->",
                           debug=False)
perceptron_p4.training()
print_plot(perceptron_p4)

# P5 = [0, -2]
P5_X = np.array([[0, -2]])
perceptron_p5 = Perceptron(values_x=P5_X,
                           values_y=[0],
                           weights=[2, 1],
                           bias=0,
                           threshold=0,
                           learning_rate=0.1,
                           activation_function=step_function,
                           message="P5 pesos finales ->",
                           debug=False)
perceptron_p5.training()
print_plot(perceptron_p5)

# P6 = [0, -2]
P6_X = np.array([[2, -2]])
perceptron_p6 = Perceptron(values_x=P6_X,
                           values_y=[0],
                           weights=[2, 1],
                           bias=0,
                           threshold=0,
                           learning_rate=0.1,
                           activation_function=step_function,
                           message="P6 pesos finales ->",
                           debug=False)
perceptron_p6.training()
print_plot(perceptron_p6)

# P7 = [0, -2]
P7_X = np.array([[2, 0]])
perceptron_p7 = Perceptron(values_x=P7_X,
                           values_y=[0],
                           weights=[2, 1],
                           bias=0,
                           threshold=0,
                           learning_rate=0.1,
                           activation_function=step_function,
                           message="P7 pesos finales ->",
                           debug=False)
perceptron_p7.training()
print_plot(perceptron_p7)

# P8 = [0, -2]
P8_X = np.array([[2, 2]])
perceptron_p8 = Perceptron(values_x=P8_X,
                           values_y=[1],
                           weights=[2, 1],
                           bias=0,
                           threshold=0,
                           learning_rate=0.1,
                           activation_function=step_function,
                           message="P8 pesos finales ->",
                           debug=False)
perceptron_p8.training()
print_plot(perceptron_p8)

plt.show()

import numpy as np

from Perceptron import step_function, Perceptron

# Se esta considerando 0 como numero par
# Para ver las iteraciones cambiar en los constructores el parametro `debug` a True


X = np.array([
    [1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 1, 0, 1],
    [1, 1, 1, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [1, 0, 1, 1, 0, 1, 1],
    [1, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 1, 1],
])

# 1 [pares] y 0 [impares]
even_Y = np.array([
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0
])

# 1 [0-5] y 0 [6-9]
gt5_Y = np.array([
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0
])

# 1 [primos] y 0 [no primos]
primes_Y = np.array([
    0,
    0,
    1,
    1,
    0,
    1,
    0,
    1,
    0,
    0
])

even_numbers_perceptron = Perceptron(values_x=X,
                                     values_y=even_Y,
                                     weights=[0, 0, 0, 0, 0, 0, 0],
                                     bias=0,
                                     threshold=0,
                                     learning_rate=0.1,
                                     activation_function=step_function,
                                     message="Pesos finales Perceptron numeros pares ->",
                                     debug=False)
even_numbers_perceptron.training()

gt5_perceptron = Perceptron(values_x=X,
                            values_y=gt5_Y,
                            weights=[0, 0, 0, 0, 0, 0, 0],
                            bias=0,
                            threshold=1,
                            learning_rate=1,
                            activation_function=step_function,
                            message="Pesos finales Perceptron mayores a 5->",
                            debug=False)
gt5_perceptron.training()

primes_perceptron = Perceptron(values_x=X,
                               values_y=primes_Y,
                               weights=[0, 0, 0, 0, 0, 0, 0],
                               bias=0,
                               threshold=0,
                               learning_rate=1,
                               activation_function=step_function,
                               message="Pesos finales Perceptron primos->",
                               debug=False)
primes_perceptron.training()

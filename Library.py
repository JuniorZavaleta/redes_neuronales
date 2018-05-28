import numpy as np

from Perceptron import step_function, Perceptron

# Se esta considerando 0 como numero par
# Para ver las iteraciones cambiar en los constructores el parametro `debug` a True


X = np.array([
    [0.7, 3],
    [1.5, 5],
    [2.0, 9],
    [0.9, 11],
    [4.2, 0],
    [2.2, 1],
    [3.6, 7],
    [4.5, 6],
])

# 1 [pares] y 0 [impares]
Y = np.array([
    [0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 1, 1, 0, 0, 1, 1]
])

W = np.array([
    [0.5, 0.5],
    [0.5, 0.5]
])

library_perceptron = Perceptron(X=X,
                                Y=Y,
                                W=W,
                                learning_rate=1,
                                activation_function=step_function,
                                message="Pesos finales libros ->",
                                debug=True,
                                bias=[0.5, 0.5],
                                threshold=0)
library_perceptron.training()

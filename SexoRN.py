import numpy as np

from Perceptron import step_function, Perceptron

X = np.array([
    [170, 56],
    [172, 63],
    [160, 50],
    [170, 63],
    [174, 66],
    [158, 55],
    [183, 80],
    [182, 70],
    [165, 54]])

Y = np.array(
    [1,
     0,
     1,
     0,
     0,
     1,
     0,
     0,
     1])
sex_perceptron = Perceptron(values_x=X,
                            values_y=Y,
                            learning_rate=0.1,
                            threshold=0,
                            weights=[0, 0],
                            activation_function=step_function,
                            bias=0.1)
sex_perceptron.training()

print("Ingresar altura 0 y peso 0 para salir")
while True:
    altura = int(input("Altura: "))
    peso = int(input("Peso: "))

    if altura == 0 and peso == 0:
        break

    if sex_perceptron.evaluate([int(altura), int(peso)]) == 0:
        print("Varon")
    else:
        print("Dama")

# Sintaxis básica de Tensorflow
import tensorflow as tf  # version 2

# Así se ejecutan las cosas en tensorflow dentro de una sesión
# En otros terminos es como un entorno virtual
with tf.compat.v1.Session() as session:

    # primer ejemplo con strings
    msj1 = tf.constant("Hola ")
    msj2 = tf.constant("Mundo")
    # Imprime un tensor de tipo string
    print('\n' + str(msj1) + '\n' + str(msj2) + '\n')
    # Imprime que es un tensor
    print('\n' + str(type(msj1)) + '\n' + str(type(msj2)) + '\n')
    # Concatenamos las cadenas y las guardamos en la variable resultado
    resultado = msj1 + msj2
    # ejecutamos el mensaje con la session
    print(session.run(resultado))
    # session.close()

    # segundo ejemplo con numeros
    a = tf.constant(10)
    b = tf.constant(5)
    # imprime un tensor de entero de 32 bits
    print('\n' + str(a) + '\n' + str(b) + '\n')
    # Concatenamos las constantes y las guardamos en la variable resultado2
    # Se ejecuta mediante la sesión y suma los valores de a + b
    resultado2 = session.run(a+b)
    # imprime el resultado de la operación matematica
    print('\nEste es el resultado de la suma de las constantes ' +
          str(resultado2) + '\n')

    # creamos una cosntante de 15 numeros
    constante = tf.constant(15)
    # una matriz de 6 x 6 con valor de 10
    matriz1 = tf.fill((6, 6), 10)
    # una matriz aleatroria de 5 x 5
    matriz2 = tf.random.normal((5, 5))
    # una matriz 4 x 4 con un valor minimo de 0 y máxino de 5
    matriz3 = tf.random.uniform((4, 4), 0, 5)
    # una matriz de zeros de 2 x 2
    matriz_ceros = tf.zeros((2, 2))
    # una matriz de unos de 3 x 3
    matriz_unos = tf.ones((3, 3))
    # Creamos una lista donde guardaos todas las variables que hemos creados(la constante y las matrices)
    operaciones = [constante, matriz1, matriz2,
                   matriz3, matriz_ceros, matriz_unos]
    # Itera cada uno de los valores con un salto de linea de cada varible que hemos creado
    for op in operaciones:
        print(session.run(op))
        print("\n")
    # Cerramos la sesión
    session.close()

# primer solucion con tensorflow v2
# sin utilizar la puta sesión de mierda de la v1
#resultado = msj1 + msj2
# tf.print(resultado)

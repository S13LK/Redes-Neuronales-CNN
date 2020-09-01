# RED NEURONAL RECURRENTE (RNN) CON TF
# Creamos una RNN con una capa de 3 neuronas, desenrollada 2 veces
# importamos las librerias
# 15/08/20
import numpy as np
import tensorflow as tf

# agregamos una sesión
with tf.compat.v1.Session() as sesion:

    # creamos las contantes
    numero_entradas = 2
    numero_neuronas = 3

    # creamos los marcados, que serán las entrada de datos a nuestras capas
    # tf de tipo float32, la forma que tendrá es fila por colulmas
    # no sabemos el numero de filas (NONE) y las columnas serán el numero_entradas
    x0 = tf.compat.v1.placeholder(tf.float32, [None, numero_entradas])
    x1 = tf.compat.v1.placeholder(tf.float32, [None, numero_entradas])

    # creamos las variables, que serán los pesos de los enlaces para nuestras capas
    # creamos numeros aleatorios, y la forma que tendrá es fila por columnas
    # en este caso, el numero de entras y numero de neuronas para WX
    # y para Wy será el numero de neuronas por el numero de neuronas
    Wx = tf.Variable(tf.random.normal(
        shape=[numero_entradas, numero_neuronas]))
    Wy = tf.Variable(tf.random.normal(
        shape=[numero_neuronas, numero_neuronas]))
    # el vias lo inicializamos con la matriz de zeros
    # y le asignamos el tamaño de una fila por el numero de neuronas
    b = tf.Variable(tf.zeros([1, numero_neuronas]))
    # creamos las funciones de salida
    # y0 será la salida de la primera capa de neuronas
    # que se calculara multiplicando x0 que es la entrada
    # por doble x que es el peso del enlace, más el vias
    # mediante la función de activación tanh
    # y como valor le pasamos la multiplicación de matrices
    # y finalmente le sumamos el vias
    y0 = tf.tanh(tf.matmul(x0, Wx) + b)
    # utilizamos la función de activación de tanh
    # y como valor le pasamos la multiplicación de matrices
    # que sería la salida de la primer capa (y0) por el peso de la segunda capa
    # más la multiplicación de matrices, que en este caso
    # sería la propia entrada de la capa 2 por el peso del enlace más el vias
    y1 = tf.tanh(tf.matmul(y0, Wy) + tf.matmul(x1, Wx) + b)

    # creamos un conjuntos de datos
    # una lista de datos con numpy array
    lote_x0 = np.array([[0, 1], [2, 3], [4, 5]])
    lote_x1 = np.array([[2, 4], [3, 9], [4, 1]])

    # creamos una variable de inicialización
    init = tf.compat.v1.global_variables_initializer()

    # ejecuta la inicialización de las variables
    sesion.run(init)
    # ejecuta la función y0 e y1 para obtener los resultados de la función de salida
    #  y los datos que necesita, osea los placeholders les pasamos mediante un diccionar los lotes que hemos creado anteriormente,
    # al ejecutarlo obtenmos dos salidas para y0 e y1

    # obtenemos dos salidas (y0, y1)
    # les asignamos 2 variables (salida_y0, salida_y1)
    # mediante la ejecución para obtener, ejecutar nuestras funcipones
    # pasandole como datos los lotes que hemos creado antes
    salida_y0, salida_y1 = sesion.run(
        [y0, y1], feed_dict={x0: lote_x0, x1: lote_x1})
    # imprime la salida de datos para y0
    print('\n' + str(salida_y0) + '\n')
    # imprime la salida de datos para y1
    print('\n' + str(salida_y1) + '\n')

    # Nota importante sobre este script:
    # implementamos mediante TF una RNN de 3 neuronas desenrollada 2 veces
    # y esta es la forma de codificarla mediante TF

# cerramos la sesión
sesion.close()

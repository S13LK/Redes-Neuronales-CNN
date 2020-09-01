# RED NEURONAL CON TENSORFLOW
# 13/08/20
# importamos la librería de tensorflow y numpy
import tensorflow as tf
import numpy as np

# Creamos la sesión
with tf.compat.v1.Session() as sesion:

    # creamos una matriz de numeros aleatorios
    # el tamaño es de 0 a 50 con 4 filas y 4 columnas
    aleatorio_a = np.random.uniform(0, 50, (4, 4))
    # imprimimos los valores de la matriz a
    print('\n' + str(aleatorio_a) + '\n')
    # creamos otra matriz de numeros aleatorios
    # el tamaño es de 0 a 50 con 4 fila y 1 columna
    aleatorio_b = np.random.uniform(0, 50, (4, 1))
    # imprimimos los valores de la matriz b
    print('\n' + str(aleatorio_b) + '\n')
    # los placeholders no tienen valores, pero si define su tipo
    a = tf.compat.v1.placeholder(tf.float32)
    b = tf.compat.v1.placeholder(tf.float32)
    # Operaciones artimeticas (Suma y multiplicación)
    # Que se mandan a ejecutar mediante la sesión.run()
    suma = a + b
    multiplicacion = a * b
    # Ejecutamos la operación de suma
    # Feed_dictionary, sirve para rellenar los valores de los placeholders
    # utilizando la operación de suma y la guardamos en la variable resultado_suma
    resultado_suma = sesion.run(suma, feed_dict={a: 10, b: 20})
    # En esta variable se guarda la suma de matrices utilizando un diccionario
    # para rellenar los placeholders (a y b) con las matrices (aleatorio_a y aleatorio_b)
    resultado_suma_matrices = sesion.run(
        suma, feed_dict={a: aleatorio_a, b: aleatorio_b})
    # Imprimimos el resultado
    print('\n' + str(resultado_suma) + '\n')
    print('\n' + str(resultado_suma_matrices) + '\n')
    # ahora lo haremos con la multiplicación
    resultado_multiplicacion = sesion.run(
        multiplicacion, feed_dict={a: 10, b: 20})
    # En esta variable se guarda la multiplicacion de matrices utilizando un diccionario
    # para rellenar los placeholders (a y b) con las matrices (aleatorio_a y aleatorio_b)
    resultado_multiplicacion_matrices = sesion.run(
        multiplicacion, feed_dict={a: aleatorio_a, b: aleatorio_b})
    # Imprimimos el resultado
    print('\n' + str(resultado_multiplicacion) + '\n')
    print('\n' + str(resultado_multiplicacion_matrices) + '\n')

    # Ejemplo de una RED NEURONAL
    # esta variable contiene 10 caracteristicas para la neurona
    caracteristicas = 10
    # Esta variable define 4 neuronas para la Neural Network
    neuronas = 4
    # Esta variable define el tipo de datos (float32) y le asignamos el tamaño (fila x columna)
    # El numero de filas suele ser el numero de datos que tenemos en nuestro juego de pruebas
    # Em este caso utilizaremos NONE que es lo normal, porque no conocemos el numero de pruebas
    # pero si conocemos las columnas que serán las caracteristicas de la Red Neuronal
    x = tf.compat.v1.placeholder(tf.float32, (None, caracteristicas))
    # Creamos una variable donde el tamaño que tiene la matriz, es la generación de numeros aleatorios con la distribución normal
    # donde el numero de filas será el numero de caracteristicas y el numero de columnas es el numero de neuronas
    w = tf.Variable(tf.random.normal([caracteristicas, neuronas]))
    # Creamos una variable donde el tamaño que tiene la matriz, es la generación de unos
    # donde la dimensión será el numero de neuronas
    b = tf.Variable(tf.ones([neuronas]))
    # creamos la multiplicación de matrices mediante TF, donde los parametros son X y W
    multiplicacion_nn = tf.matmul(x, w)
    # Esta es la suma de la multiplicación
    z = tf.add(multiplicacion_nn, b)
    # creamos la función de activación mediante TF, para conseguir el resultado final de la neurona
    # mediante la funcion de activación sigmoid donde le pasamos el valor de Z
    activación = tf.sigmoid(z)
    # creamos una variable para inicializamos todas la variables
    inicializacion = tf.compat.v1.global_variables_initializer()
    # creamos una variable, que almacena una matriz de numeros aleatorios
    # su dimensión es de una fila por las caracteristicas
    valores_x = np.random.random([1, caracteristicas])
    # imprimimos los valores
    print('\n' + str(valores_x) + '\n')
    # ejecutamos la inicialización de todas las variables mediante la sesión
    sesion.run(inicializacion)
    # creamos una variable donde se almacena el resultado de la red neuronal
    # ejecuta la función de activación y el diccionario rellena los datos del placeholder
    # para que genere los resultados de la función de activación
    resultado_nn = sesion.run(activación, feed_dict={x: valores_x})
    # Imprimimos el resultado de la red neuronal
    print('\n' + str(resultado_nn) + '\n')

    # Cerramos la sesión
    sesion.close()

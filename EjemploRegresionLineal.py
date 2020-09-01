# EJEMPLO DE REGRESION LINEAL CON TF
# 13/08/20
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Creamos una sesion
with tf.compat.v1.Session() as sesion:

    # Generamos mediante numpy una secuencia de numeros que van del 0 al 100, 10 elementos
    # y a cada uno de esos 10 elementos le vamos a sumar un numero aleatorio que va de - 1 a 1
    # una especie de ruido que le vamos añadir a linespace, es para que no sean tan uniformes los numeros
    datos_x = np.linspace(0, 10, 10) + np.random.uniform(-1, 1, 10)
    # imprimimos los valores que hemos generados para X
    print('\n' + str(datos_x) + '\n')
    # Hace lo mismo que datos_X
    datos_y = np.linspace(0, 10, 10) + np.random.uniform(-1, 1, 10)
    # imprimimos los valores que hemos generado para Y
    print('\n' + str(datos_y) + '\n')
    # creamos el grafico que contiene datos_X y datos_Y
    # el * lo utilizamos para marcar los puntos que genera
    plt.plot(datos_x, datos_y, '*')
    # esta linea sirve para mostrar el grafico en una ventana emergente
    # plt.show()

    # crear una red neuronal
    # que resulva la ecuación (Y)
    #y = mx + b
    # imprimimos los numeros aleatorios
    print('\n' + str(np.random.rand(2)) + '\n')
    # le asignamos los 2 numeros aleatorios a cada variable (m y b)
    # solo para llevar a cabo este ejemplo de regresión lineal
    m = tf.Variable(0.73)
    b = tf.Variable(0.78)
    # iniciamos una variable a 0
    error = 0
    # por cada uno de los elementos (X,Y) que tenemos en nuestros datos (X e Y)
    for x, y in zip(datos_x, datos_y):
        # calculamos cual nos va a salvar el valor de predicción de Y
        y_pred = m * x + b  # En función de esta ecuación
        # error es igual a el error más Y menos Y_pred elevado al cuadrado
        # Genera un error entre lo que es el valor real y el valor de predicción
        error = error + (y - y_pred)**2

    # ahora optimizaremos o utilizar una función de TF
    # para disminuir al máximo el error y obtener una regresion lineal lo más correcta posible
    # esta variable almacena mediante TF, el entrenamiento con GradientDescentOptimizer
    # el parametro es la taza de aprendizaje de un valor X
    optimizador = tf.compat.v1.train.GradientDescentOptimizer(
        learning_rate=0.001)
    # esta variable de entrenamiento a intentado o está intentado minimizar o reducir al máximo el error
    entrenamiento = optimizador.minimize(error)
    # inicializamos todas las variables
    inicialización = tf.compat.v1.global_variables_initializer()
    # mediante la sesión ejecutamos la inicialización de todas las variables
    sesion.run(inicialización)
    # esta variable sirve para condicionar nuestro bucle
    pasos = 1
    # y nuestro bucle se ejecutara sólo una vez
    for i in range(pasos):
        # mediante la sesion ejecutamos el entrenamiento que es optimizar el error
        sesion.run(entrenamiento)
        # obtenemos valores de M y B que hemos obtenido mediante con el optimizador
        final_m, final_b = sesion.run([m, b])

    # ahora calcularemos la grafica con los valores M y B
    # esta variable contiene 10 valores entre -1 y 11
    x_test = np.linspace(-1, 11, 10)
    # esta variable almacena nuevos valores de predicción, con los nuevos valores de M y B
    y_pred_2 = (final_m * x_test) + final_b
    # creamos  un grafico con los nuevos valores de predicción
    plt.plot(x_test, y_pred_2, 'r')
    # mostramos los puntos que hemos creado inicialmente con la primer grafica
    plt.plot(datos_x, datos_y, 'b*')
    # mostramos la grafica en una ventana emergente
    plt.show()
# y cerramos la sesión de este ejemplo
sesion.close()

# Variables y Placeholders
# 13/08/20
# Impotamos la librería de tensorflow
# y le asignamos un alias "tf"
import tensorflow as tf

# Así se ejecutan las cosas en tensorflow dentro de una sesión
# En otros terminos es como un entorno virtual
with tf.compat.v1.Session() as sesion:

    # imprime la versión de tf
    print('\nVersion actual de TF ' + str(tf.__version__) + '\n')

    # las variables sirven para almacenar datos
    # como por ejemplo los pesos en los enlaces entre las neuronas
    # y también el valor via de cada uno

    # esta es la manera correcta de crear una variable de tipo random.uniform
    # en esta variable guardamos una matriz de 5x5 con un valor minimo y un valor máximo
    tensor = tf.random.uniform((5, 5), 0, 1)
    # creamos una variable que será igual a un tensor
    # y le pasamos el valor inicial que es el tensor que creamos primero
    variable = tf.Variable(initial_value=tensor)
    # imprimimos el valor de la variable
    print('\n Aqui ejecutamos la variable que hemos inicializado como tensor ' +
          str(variable) + '\n')
    # las variable siempre necesitan ser inicializadas
    # así que creamos una variable para inicializar las variables globales
    # con la V1 de tensorflow se ejecuta de esta manera
    inicializador = tf.compat.v1.global_variables_initializer()
    # o también se puede ejecutar de la siguiente manera, aun que la primera y segunda opción están obsoleta desde el 2017
    #i = tf.compat.v1.initialize_all_variables()
    # ejecutamos la sesión eh inicializamos las variables globales
    sesion.run(inicializador)
    # guardamos en la variable resultado, el valor de la variable mediantes la sesion
    # la cual es una variable de tipo tensor
    resultado = sesion.run(variable)
    # imprimimos el resultado de la variable
    print('\n' + str(resultado) + '\n')
    # Placeholders están inicialmente vacios
    # Se utilizan para alimentar los ejemplos de entranamiento del modelo
    # Una especie como de incognita dentro de las ecuaciones
    # Se ejecuta la versión1 para poder hacer uso del placerholder
    # Ya que la versión2 no ocupa placeholders
    # tf.placeholder(tf.float32,shape=(20,20))
    incognitas = tf.compat.v1.placeholder(tf.float32, (20, 20))
    # imprimimos el valor de incognitas
    print('\n' + str(incognitas) + '\n')
# Cerramos la sesión
sesion.close()

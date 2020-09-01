# Ejemplo de Red Neuronal Recurrente (RNN) - Series Temporales
# Creamos
# importamos las librerias
# 15/08/20
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import keras as ke

# almacena la ruta del archivo
ruta = 'D:/Python Projects/TensorFlowProject/Excel-Ejemplo/produccion-leche.csv'
# almacena un dataframe, mediante pandas lee el archivo CSV
leche = pd.read_csv(ruta, index_col='Month')
# imprime los primeros 5 elementos de nuestro DataFrame
print('\n' + str(leche.head()) + '\n')
# imprime más información sobre el dataframe
print('\n' + str(leche.info()) + '\n')
# creamos un indice para la serie de tiempo con pandas
leche.index = pd.to_datetime(leche.index)
# creamos una grafica mediante el metodo plot
# para visualizar los datos en función del tiempo
leche.plot()
# muestra el grafico
# plt.show()
# ahora dividimos estas 168 entradas de datos
# para el conjunto de entrenamiento le pasamos mediante el metodo head los primero 150 datos
conjunto_entrenamiento = leche.head(150)
# para el conjunto de pruebas le pasamos mediante el metodo tail los ultimos 18 datos
conjunto_pruebas = leche.tail(18)
# imprime los conjuntos de datos para entrenamientos y pruebas
print('\n' + str(conjunto_entrenamiento) + '\n')
print('\n' + str(conjunto_pruebas) + '\n')
# ahora normalizaremos los datos para poder utilizar el TF
normalizacion = MinMaxScaler()
# ahora creamos unos datos de entrenamiento normalizados, mediante el fit_transform
entrenamiento_normalizados = normalizacion.fit_transform(
    conjunto_entrenamiento)
# y también normalizamos los datos de prueba, mediante el fit_transform
pruebas_normalizadas = normalizacion.fit_transform(conjunto_pruebas)
# imprime los datos normalizados, para ambos casos (entrrenamiento y pruebas)
print('\n' + str(entrenamiento_normalizados) + '\n')
print('\n' + str(pruebas_normalizadas) + '\n')
# creamos una función para crear lotes de datos (entrenamiento y pruebas)
# como parametro, le ponemos los datos de entrenamiento, tamaño de lote y el numero de pasos


def lotes(datos_entrenamiento, tamaño_lote, pasos):
    # definimos el comienzo con esta variable qué
    # crea un numero aleatorio, entre 0 y la longitud de los datos de entrenamiento menos los pasos
    comienzo = np.random.randint(0, len(datos_entrenamiento) - pasos)
    # indexar los datos desde el comienzo
    # como parametros, le pasamos los datos de entrenamiento
    # y los indexaremos desde el comienzo hasta el comienzo más pasos más uno
    # lo redimencionamos con reshape, uno, pasos más uno
    lote_y = np.array(
        datos_entrenamiento[comienzo:comienzo+pasos+1]).reshape(1, pasos+1)
    # devolvemos los lotes de datos redimencionados
    return lote_y[:, :-1].reshape[-1, pasos, 1], lote_y[:1:].reshape(-1, pasos, 1)

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Segunda parte
# Definimos las constantes que vamos a utilizar
# Nota importante: al parecer contrib ya esta en deshuso y hay que migrar a TF V2


numero_entradas = 1
numero_pasos = 18
numero_neuronas = 120
numero_salidas = 1
taza_aprendizaje = 0.001
numero_iteracciones_entrenamiento = 5000
tamaño_lote = 1

# Ahora definimos los placeholders

x = tf.compat.v1.placeholder(tf.float32, [None, numero_pasos, numero_entradas])
y = tf.compat.v1.placeholder(tf.float32, [None, numero_pasos, numero_salidas])
# Ahora creamos la capa de la red neuronal recurrente
# utilizamos TF para definirla, mediante contrib, rnn, OutputProjectionWrapper
# y le pasamos como parametros, TF, mediante contrib, rnn, BasicLSTMCell
# y le pasamos como parametros, el numero de units que es el numero de neuronas
# y le pasamos activation que es igual a TF, NN, RELU como función de activación (cerramos parentesis)
# y el ultimo parametro de la función inicial es el output_size que es el numero de salidas
capa = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicLSTMCell(
    num_units=numero_neuronas, activation=tf.nn.relu), output_size=numero_salidas)
# definimos las variables de salidas y estados
salidas, estados = tf.nn.dynamic_rnn(capa, x, dtype=tf.float32)
# creamos la función de coste, la función de error
funcion_error = tf.reduce_mean(tf.square(salidas-y))
# creamos el optimizador
optimizador = tf.train.AdamOptimizer(learning_rats=taza_aprendizaje)
# creamos el entrenamiento
entrenamiento = optimizador.minimize(funcion_error)
# inicia las variables globales
init = tf.compat.v1.global_variables_initializer()
# no se que haga pero lo investigare
saver = tf.train.Saver()

# ejecutamos la sesión de TF
with tf.compat.v1.Session() as sesion:
    # inicializamos las variables
    sesion.run(init)
    # creamos un bucle
    # en el cual haremos una iteracción de 5000 veces
    for iteraccion in range(numero_iteracciones_entrenamiento):
        # generamos en cada iteracción lote_x y lote_y
        lote_x, lote_y = lotes(entrenamiento_normalizados,
                               tamaño_lote, numero_pasos)
        # ejecutamos el entrenamiento para minimizar el error
        sesion.run(entrenamiento, feed_dict={x: lote_x, y: lote_y})
        # preguntamos si la iteracción es un multiplo de 100
        if iteraccion % 100 == 0:
            # para calcular el error que tenemos
            error = funcion_error.eval(feed_dict={x: lote_x, y: lote_y})
            # y lo imprimimos
            print(iteraccion, '\t Error ', error)
        # salva la sesión en la siguiente ruta
        # hay que crear la carpeta para guardar el entrenamiento
        saver.save(sesion, './modelo_series_temporales')

    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    # Tercera parte
    # Predecimos el futuro
    # hay que generas las predicciones
    # recuperamos la sesión de antes con el fichero que creamos anteriormente
    saver.restore(sesion, '')
    # generamos una variable, donde le asignamos una lista
    # sobre los datos de entrenamiento normalizado
    entrenamiento_seed = list(entrenamiento_normalizados[-18])
    # creamos un bucle, donde generamos lotes_x para generar las predicciones de estos 18 elementos de pruebas
    for iteraccion in range(18):
        # generamos el lote_x, le pasamos el entrenamiento seed
        # y lo redimencionamos
        lote_x = np.array(
            entrenamiento_seed[-numero_pasos]).reshape(1, numero_pasos, 1)
        # creamos la predicción, y ejecutamos la salida
        # pasando los datos del placeholder que se van a generar
        prediccion_y = session.run(salidas, feed_dict={x: lote_x})
        # añadimos la predicción a nuestra variable entrenamiento seed
        entrenamiento_seed.append(prediccion_y[0, -1, 0])
    # obtenemos los resultados definitivos, mediante la normalización
    # con el metodo inverse_transform, y le pasamos los datos de entrenamiento seed
    # y por ultimo lo redimencionamos a 18 filas una columna
    resultados = normalizacion.inverse_transform(np.array(entrenamiento_seed[18:]).reshape=(18, 1))
    # imprime los resultados
    print('\n' + str(resultados) + '\n')
    # al conjunt de pruebas le vamos añadir otra columna con nuestras estimaciones
    conjunto_pruebas['Predicciones'] = resultados
    # imprime el conjunto de pruebas con nuestras estimaciones
    print('\n' + str(conjunto_pruebas) + '\n')
    # visualizamos el grafico
    conjunto_pruebas.plot()

# cerramos la sesión
sesion.close()

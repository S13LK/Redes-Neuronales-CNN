# Predecir una imagen de moda
# 17/08/2020

# importamos TF y Keras
import tensorflow as tf
from tensorflow import keras

# librerias de ayuda
import numpy as np
import matplotlib.pyplot as plt

# imprime la versión de TF
print('\n' + str(tf.__version__) + '\n')

# almacenamos el dataset en esta variable
mnist = tf.keras.datasets.fashion_mnist
# importamos el set de datos de moda de MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# convierte los ejemplos de numeros enteros a numeros de punto flotante
x_train, x_test = x_train / 255.0, x_test / 255.0

# contruye un modelo aplicando capas
# asignamos un optimizador y una función de perdida
# para el entrenamiento del modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# entrena y evalua el modelo
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)

# el modelo de clasificacion de imagenes
# fue entrenado y alcanzo una exactitud de 98%
# en este conjunto de datos, como conclusión
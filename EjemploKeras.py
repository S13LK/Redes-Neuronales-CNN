# Ejemplo con Keras
# 17/08/2020
# Importamos las librerias
import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import models, layers, losses, optimizers, metrics, activations
from sklearn.metrics import classification_report

# vino almacena el metodo load_wine()
vino = load_wine()
# imprimimos el valor del dataset
print(vino['DESCR'])
# obtenemos las columnas caracteristicas y objectivo
caracteristicas = vino['data']
objectivo = vino['target']
# # dividimos los datos (entrenamiento y test)
x_train, x_test, y_train, y_test = train_test_split(
    caracteristicas, objectivo, test_size=0.3)
# creamos la variable normalizador y le asignmaos el metodo de Sklearn Preprocesing
normalizador = MinMaxScaler()
# normalizamos los datos
x_train_normalizado = normalizador.fit_transform(x_train)
x_test_normalizado = normalizador.transform(x_test)

# empezamos con keras perros
# creamos el modelo que vamos a entrenar
modelo = models.Sequential()
# añadimos las capas de la red neuronal
# en este caso sería la capa de entrada principal
modelo.add(layers.Dense(units=13, input_dim=13, activation='relu'))
# añadimos 2 capas intermedias
modelo.add(layers.Dense(units=13, activation='relu'))
modelo.add(layers.Dense(units=13, activation='relu'))
# por ultimo creamos la capa de salida
modelo.add(layers.Dense(units=3, activation='softmax'))
# compilamos el modelo, y como parametos utilizamos los siguientes:
# optimizer = el optimizador que se utlizo
# loss = función de perdida de error
# metrics = validación del modelo, precision
modelo.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# entrenamos el modelo 60 veces
modelo.fit(x_train_normalizado, y_train, epochs=60)
# una vez el modelo este entrenado, podemos calcular las predicciones
predicciones = modelo.predict_classes(x_test_normalizado)
# evaluamos el modelo para ver nuestra precisión de datos
print('\n' + str(classification_report(y_test, predicciones)) + '\n')

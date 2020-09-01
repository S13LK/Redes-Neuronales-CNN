# Ejemplo con API Estimator
# 17/08/2020
# importamos las librerias
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import estimator
from sklearn.datasets import load_wine  # datos de vinos
from tensorflow import keras

# vino almacena el metodo load_wine()
vino = load_wine()
# imprimimos el valor del dataset
print(vino['DESCR'])
# Obtenemos las caracteristicas y el objectivo
caracteristicas = vino['data']
objectivo = vino['target']
# dividimos los datos (entrenamiento y test)
x_train, x_test, y_train, y_test = train_test_split(
    caracteristicas, objectivo, test_size=0.3)
# creamos la variable normalizador y le asignmaos el metodo de Sklearn Preprocesing
normalizador = MinMaxScaler()
# normalizamos los datos
x_train_normalizado = normalizador.fit_transform(x_train)
x_test_normalizado = normalizador.transform(x_test)
# imprime los valores normalizados
print('\n' + str(x_train_normalizado) + '\n')
print('\n' + str(x_test_normalizado) + '\n')
# imprime filas y comulmas de entrenamiento y pruebas
print('\n' + str(x_train_normalizado.shape) + '\n')
# print('\n' + str(x_test_normalizado.shape) + '\n')
# calcula las caracteristicas de las columnas
columnas_caracteristicas = [tf.feature_column.numeric_column('x', shape=[13])]
# creamos un modelo, mediante la api de estimator usando el DNNClassifier
modelo = estimator.DNNClassifier(hidden_units=[20, 20, 20], feature_columns=columnas_caracteristicas,
                                 n_classes=3, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
# tf.train.GradientDescentOptimizer(learning_rate=0.01))
# ahora entrenaremos nuestro modelo
# pero primero creamos nuestra función de entrada
funcion_entrada = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={'x': x_train_normalizado}, y=y_train, shuffle=True, batch_size=10, num_epochs=10)
# funcion original, pero no funciona con TF v2
# estimator.inputs.numpy_input_fn(x={'x': x_train_normalizado}, y=y_train, shuffle=True, batch_size=10, num_epochs=10)
# ejecuta el entrenamiento 600 veces
modelo.train(input_fn=funcion_entrada, steps=600)
# una vez el modelo este entrenado, podemos calcular las predicciones
# mediante la funcion de evaluación
funcion_evaluacion = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={'x': x_test_normalizado}, shuffle=False)
# guardamos las predicciones
predicciones = list(modelo.predict(input_fn=funcion_evaluacion))
# calculamos las predicciones finales y las guardamos en una lista
# estas prediciones finales serán los resultados de Y
predicciones_finales = [p['class_ids'][0] for p in predicciones]
# imprime las predicciones finales
print('\n' + str(classification_report(y_test, predicciones_finales) + '\n'))
# ahora comparamos los valores reales con los de test

# Ejemplo de regresión con TF
# 14/08/20
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

ruta = "D:\Python Projects\TensorFlowProject\Excel-Ejemplo\precios_casas.csv"
# creamos un dataframe para el archivo CSV
casas = pd.read_csv(ruta)
# imprime los primeros 5 valores
print('\nDatos Completos CSV\n\n' + str(casas.head()) + '\n')

# Separamos el dataframe en 2 partes
# Es decir, le quitamos la columna de median_house_value
# Y le mencionamos el eje que es el valor de una columna y no de una fila
casas_x = casas.drop('median_house_value', axis=1)

# imprime el dataframe sin la columna median_house_value
print('\nDatos sin columna MHV\n\n' + str(casas_x.head()) + '\n')

# almacena la columna que borramos de casas_x
casas_y = casas['median_house_value']

# Imprime los valores de casas_y
print('\nDatos columna MHV\n\n' + str(casas_y.head()) + '\n')

# datos_x y datos_y los tenemos que dividir para datos de entrenamiento
x_train, x_test, y_train, y_test = train_test_split(
    casas_x, casas_y, test_size=0.30)

# verifiacamos la división de train y test
print('\nX_TRAIN\n\n' + str(x_train.head()) + '\n')
print('\nX_TEST\n\n' + str(x_test.head()) + '\n')
print('\nY_TRAIN\n\n' + str(y_train.head()) + '\n')
print('\nY_TEST\n\n' + str(y_test.head()) + '\n')

# ahora normalizeremos los datos para poder utilizarlos con tensorflow
normalizador = MinMaxScaler()
# esta variable sirve para entrenar FIT el modelo mediante los datos de entrenamiento
normalizador.fit(x_train)
# imprime el normalizador
print('\nNormalizador\n\n' + str(normalizador) + '\n')
# sobrescribimos la variable x_train
# mediante el normalizador vamos a transformar estos datos de entrenamiento
# para que los datos de ahora de x_train esten normalizados
# entre el 0 y 1
x_train = pd.DataFrame(data=normalizador.transform(
    x_train), columns=x_train.columns, index=x_train.index)
# imprime los datos de x_train normalizados
print('\nX_TRAIN NORMALIZADOS\n\n' + str(x_train) + '\n')
# hacemos lo mismo con la variable x_test
x_test = pd.DataFrame(data=normalizador.transform(
    x_test), columns=x_test.columns, index=x_test.index)
# imprime los datos de x_test normalizados
print('\nX_TEST NORMALIZADOS\n\n' + str(x_test) + '\n')
# verificamos la columnas que existen en nuestro dataframe (casas)
print('\nColumnas DataFrame\n\n' + str(casas.columns) + '\n')
# Ahora vamos a crear las variables con las columnas de categorias
longitude = tf.feature_column.numeric_column('longitude')
latitude = tf.feature_column.numeric_column('latitude')
housing_median_age = tf.feature_column.numeric_column('housing_median_age')
total_rooms = tf.feature_column.numeric_column('total_rooms')
total_bedrooms = tf.feature_column.numeric_column('total_bedrooms')
population = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
median_income = tf.feature_column.numeric_column('median_income')
# creamos una lista que almacena todas las variables que hemos creado
# para las columnas de nuestro archivo CSV
columnas = [longitude, latitude, housing_median_age, total_rooms,
            total_bedrooms, population, households, median_income]
# creamos una función de entrada, donde le hemos pasado los datos de x_train y de y_train que hemos calculado anteriormente
funcion_entrada = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=x_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)
# Creamos el modelo mediante DNNRegressor, el numero de capas ocultas es de 3
# y cada una con 10 nodos, y vamos a utilizar las columnas de catehorias que hemos creado anteriormente
modelo = tf.compat.v1.estimator.DNNRegressor(
    hidden_units=[10, 10, 10], feature_columns=columnas)
# Entrenamos el modelo, mediante la función de entrada que hemos creado
# y ejecutamos el entrenamiento 8 mil veces
modelo.train(input_fn=funcion_entrada, steps=8000)

# ahora vamos a generar las predicciones
# creamos una función de entrada para la predicción
# donde los valores de x serán los valores de pruebas
# no pasaremos ningun valor de y porque eso es lo que queremos predecir
funcion_entrada_predicciones = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=x_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
# creamos el generador de predicciones
# utilizando el modelo y el metodo predict()
generador_predicciones = modelo.predict(funcion_entrada_predicciones)
# creamos una lista de predicciones
# mediante el generador de predicciónes que acabamos de calcular
predicciones = list(generador_predicciones)
# imprime las predicciones
print('\nPredicciones\n\n' + str(predicciones) + '\n')
# creamos una lista vacia de predicciones finales
# creamos un bucle para recoger los valores que hemos estimado
# como el precio medio de las casas
predicciones_finales = []
# creamos un bucle, el cual por cada predicción dentro de predicciones
# recogemos las predicciones finales; y añadimos a la lista vacia "predicción"
# y al final recogemos el valor de predictions
for prediccion in predicciones:
    predicciones_finales.append(prediccion['predictions'])
# imprime las predicciones finales
print('\nPredicciones Finales\n\n' + str(predicciones_finales) + '\n')
# calculamos el error paratico medio
# y lo utilizamos, mediante los valores reales de test
# y las prediciones finales que hemos calculado mediante nuestro modelo
# para que las puede calcular y el resultado lo multiplicamos por 0.5
resultado = mean_squared_error(y_test, predicciones_finales)**0.5
# imprime el resultado, el cual es el error paratico medio
print('\nResultado\n\n' + str(resultado) + '\n')


# Ejemplo de clasificación con tensorflow
# Intentar predicir los ingresos de una persona en función de sus caracteristicas
# 13/08/20
# importamos las librerias
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd

# esta variable almacena la ruta del archivo CSV
ruta = "D:\Python Projects\TensorFlowProject\Excel-Ejemplo\ingresos.csv"
# en esta variable almacenamos y leemos el archivo CSV
ingresos = pd.read_csv(ruta)
# imprime el archivo CSV
print('\n' + str(ingresos) + '\n')
# el metodo unique nos devuelve una lista o un array
# de los elementos diferentes que hay dentro de esta columna (income)
print('\n' + str(ingresos['income'].unique()) + '\n')

# creamos una función
# esta función cambia el valor de la columna, ya sea (1 o 0)
# 1 para los que ganan <= 50k
# 0 para los que ganas > 50k


def cambio_valor(valor):
    if valor == '<=50K':
        return 0
    else:
        return 1


# al dataframe ingresos, lo modificamos para ponerle el valor de la columna
# pero aplicandole una función, la cual es la que ya hemos creado anteriormente
ingresos['income'] = ingresos['income'].apply(cambio_valor)
# imprime el cambio que se acaba de realizar en la función
# el metodo head se utiliza para devolver las filas superiores n
# que en este caso son los primeros 5 elementos
print('\n' + str(ingresos.head()) + '\n')
# genera los datos X que serán las caracteristicas de nuestro cojunto de datos,
# excepto la columna que queremos predecir, borrando la columna income,
# y le decimos que lo haga sobre las columnas, poniendo axis = 1 (borra la columna income)
datos_x = ingresos.drop('income', axis=1)
# imprime los valores sin la columna income
print('\n' + str(datos_x.head()) + '\n')
# los datos objectivos que queremos que predecir, en este caso serán solamente la columna (income)
# y en esta variable guardamos los datos de la fila income
datos_y = ingresos['income']
# imprime los datos de income
print('\n' + str(datos_y) + '\n')
# divide los valores de X y Y
# asigna el 70% en el train de X y Y
# y el 30% en el test de X y Y
# Es un división de un conjunto de datos
x_train, x_test, y_train, y_test = train_test_split(
    datos_x, datos_y, test_size=0.3)
# imprime los valores de entrenamiento y test
print('\n' + str(x_train.head()) + '\n')
print('\n' + str(x_test.head()) + '\n')
# vemos que columnas tenemos en nuestro dataframe
print('\n' + str(ingresos.columns) + '\n')
# creamos las variables que van almacenar cada una de las columnas
# utilizamos tf, caracteristica de columna y utilizamos vocabulary list
# ya que sabemos que valores ocupamos para esta columna, ya sea mujer o hombre
gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", [
                                                                   'Female, Male'])
# utilizamos tf, caracteristica de columna y utilizamos hash bucket
# cuando no sabemos el numero de elementos que hay se utiliza este tipo para cadenas
occupation = tf.feature_column.categorical_column_with_hash_bucket(
    "occupation", hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket(
    "marital-status", hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket(
    "relationship", hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket(
    "education", hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket(
    "native-country", hash_bucket_size=1000)
workclass = tf.feature_column.categorical_column_with_hash_bucket(
    "workclass", hash_bucket_size=1000)
# ahora haremos lo mismo con las variables que son de tipo númerico
# pero utilizando otro tipo de caracteristica para la columna de tipo númerico
age = tf.feature_column.numeric_column("age")
fnlwgt = tf.feature_column.numeric_column("fnlwgt")
educational_num = tf.feature_column.numeric_column("educational-num")
capital_gain = tf.feature_column.numeric_column("capital-gain")
capital_loss = tf.feature_column.numeric_column("capital-loss")
hours_per_week = tf.feature_column.numeric_column("hours-per-week")
# creamos una lista y en esta lista guardaremos
# todas las variables que hemos creado para nuestras columnas del archivo CSV
columnas_categorias = [gender, occupation, marital_status, relationship, education, native_country,
                       workclass, age, fnlwgt, educational_num, capital_gain, capital_loss, hours_per_week]
# creamos una función de entrada que vamos a usar para la estimación
# le hemos pasado los datos de entrenamiento de las x y las soluciones
# es decir, los datos reales que tendríamos que estimar, simplemente para entrenar
# para que haga una función de entrada para estimador
# utilizando la versión 1 de tensorflow, aunque la consola nos dice que se ocupe tf.data
funcion_entrada = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=x_train, y=y_train, batch_size=100, num_epochs=None, shuffle=True)
# creamos un modelo mediante TF con un estimator de clasificador lineal
# que tiene un atributo feature_colums que será nuestrsd columnas categorias
modelo = tf.compat.v1.estimator.LinearClassifier(
    feature_columns=columnas_categorias)
# entrana nuestro modelo, la función de entrada es la función que hemos creado
# y vamos a ejecutar el entrenamiento 8 mil veces
modelo.train(input_fn=funcion_entrada, steps=8000)

# 14/08/20 -*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*-

# Desde este punto empezaremos mañana para terminar este proceso
# una vez entrenado el modelo, haremos una función de predicción
# para predecir el valor de Y, en esta caso del conjunto de datos de X_Test
# utilizamos batch_size como parametro, que es la longitud de los datos
funcion_prediccion = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=x_test, batch_size=len(x_test), shuffle=False)
# creamos un generador de predicciones a partir del modelo
# mediante el metodo predic, y le pasamos como parametro la función de predicción que hemos creado anteriormente
generador_predicciones = modelo.predict(input_fn=funcion_prediccion)
# creamos una lista de predicciones a partir del generador
# al visualizar las predicciones, nos damos cuenta que ha creado una clase
# el valor que nos interesa es el class_ids
predicciones = list(generador_predicciones)
# imprimimos la clase predicciones que se ha generado mediante el generador de predicciones
# print('\nLista' + str(predicciones) + '\n')
# recogemos el valor de class_ids, el elemento 0
# dentro de un bucle predicción dentro de predicciones
predicciones_finales = [prediccion['class_ids'][0]
                        for prediccion in predicciones]
# imprime las predicciones finales
# print('\n Predicciones Finales' + str(predicciones_finales) + '\n')

# genera un informe de clasificación
# utilizando sklearn.metric import clasification_report
print(classification_report(y_test, predicciones_finales))

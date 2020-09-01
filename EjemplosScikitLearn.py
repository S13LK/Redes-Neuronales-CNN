# Dividir el conjunto de datos disponible, en datos para pruebas y en datos para test
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# creamos unos datos aletorios del 0 al 100
# y lo organizamos en 100 filas por 4 columnas
datos = np.random.randint(0, 100, (100, 4))
print('\n' + str(datos) + "\n")

# creamos un dataframe
dataframe = pd.DataFrame(data=datos, columns=['c1', 'c2', 'c3', 'etiqueta'])
print('\n' + str(dataframe) + "\n")

# obtenemos las caracteristicas de las primeras 3 columnas
x = dataframe[['c1', 'c2', 'c3']]
print('\n' + str(x) + "\n")

# obtenemos la caracteristica solo de la etiqueta
y = dataframe[['etiqueta']]
print('\n' + str(y) + "\n")

# divide los valores de X y Y
# asigna el 70% en el train de X y Y
# y el 30% en el test de X y Y
# Es un división de un conjunto de datos
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print('\n' + str(x_train.shape) + "\n")
print('\n' + str(x_test.shape) + "\n")
print('\n' + str(y_train.shape) + "\n")
print('\n' + str(y_test.shape) + "\n")

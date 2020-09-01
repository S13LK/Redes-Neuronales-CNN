import pandas as pd  # impotamos la libreria de pandas
# Esta variable la ocupamos para guardar la ruta del CSV
ruta = 'D:\Python Projects\TensorFlowProject\Excel-Ejemplo\personas.csv'
# creamos un dataframa para leer los archivos de Excel
dataframe = pd.read_csv(ruta)
# esta variable la ocupamos para filtrar la condición del salario mayor a 30,000
filtro = dataframe['SALARIO'] > 30000
print('\n' + str(dataframe) + "\n")  # imprimo los datos en consola
# información del dataframe, una serie de datos estadisticos
print('\n' + str(dataframe.describe()) + "\n")
# obtenemos todos los datos de la columna NOMBRE
print('\n' + str(dataframe['NOMBRE']) + "\n")
# obtenemos un dato en especifico de la primer columna
print('\n' + str(dataframe['NOMBRE'][0]) + "\n")
# seleccionamos más de una columna, y lo guardamos en una lista
print('\n' + str(dataframe[['NOMBRE', 'APELLIDOS']]) + "\n")
# obtenemos true or false del salario mayor a 30,000
print('\n' + str(filtro) + "\n")

# creamos otra dataframa con los datos del primer dataframe
# y le agregamos el filtro, para que nos muestre los datos que cumplen con la condición
dataframe2 = dataframe[filtro]
print('\n' + str(dataframe2) + "\n")  # imprimimos los resultados

# convertimos el segundo dataframe en un arreglo
# para poder realizar operaciónes
arraydf2 = dataframe2.values
print("\n" + str(arraydf2) + '\n')  # obtenemos el arreglo
# accedemos a datos especificos del arreglo
# en este caso sería al nombre, pero se puede hacer con cualquier dato
# dependiendo la posición en donde se encuentre
print("\n" + str(arraydf2[0, 0]) + '\n')

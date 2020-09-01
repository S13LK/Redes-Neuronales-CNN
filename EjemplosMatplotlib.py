# visualización de datos
import numpy as np  # importamos la libreria de numpy
import pandas as pd  # importamos la libreria de pandas
import matplotlib.pyplot as plt  # importamos la libreria de matplotlib

# en esta variable guardamos un rango de numeros del 0 al 20
x = np.arange(0, 20)
print('\n' + str(x) + "\n")

# esta variable tiene los mismos valores que x pero al cuadrado
y = x**2
print('\n' + str(y) + "\n")

# unimos los valores de X/Y y le asignamos un color a la grafica
# tambien podemos cambiar el tipo de grafico
#grafica = plt.plot(x,y,'r*')
# agregramos un titulo
plt.title("Mi Primer Grafico con Matplotlib")
# agregamos etiquetas para x or y
plt.xlabel("Eje de los valores X")
plt.ylabel("Eje de los valores Y")
# imprime la grafica como un objecto
print('\n' + str(plt.plot(x, y, 'r--')) + "\n")
# muestra la grafica
print('\n' + str(plt.show()) + "\n")


# creamos un arreglo con un rango de numeros del 0 al 50
# la forma sera de 10 filas por 5 columnas
arreglo = np.arange(0, 50).reshape(10, 5)
print('\n' + str(arreglo) + "\n")

# creamos un grafico de imagen
# convierte el valor numerico en un color
print('\n' + str(plt.imshow(arreglo)) + "\n")
# creamos una barra de color para ver las relaciones de los colores
print('\n' + str(plt.colorbar()) + "\n")
print('\n' + str(plt.show()) + "\n")


# creamos otro arreglo con valores aleatorios y no secuencial
arreglo2 = np.random.randint(0, 1000, 100)
print('\n' + str(arreglo2) + "\n")
# clasificamos los valores 10 filas y 10 columnas
# organiza los datos 10x10
clasificado = arreglo2.reshape(10, 10)
print('\n' + str(clasificado) + "\n")
# creamos el grafico de imagen con el segundo arreglo
#print('\n' + str(plt.imshow(arreglo2)) + "\n")
plt.imshow(clasificado)
# creamos una barra de color para ver las relaciones de los colores
#print('\n' + str(plt.colorbar()) + "\n")
plt.colorbar()
#print('\n' + str(plt.show()) + "\n")
# mostramos el histograma
plt.show()


# En este ejemplo utilizaremos el archivo CSV para crear un grafico

ruta = 'D:\Python Projects\TensorFlowProject\Excel-Ejemplo\personas.csv'
dataframe = pd.read_csv(ruta)

# hacer un grafico que relacione el salario con la edad
dataframe.plot(x='SALARIO', y='EDAD', kind='bar')
plt.show()
# Este es otro ejemplo con un grafico diferente
dataframe.plot(x='SALARIO', y='EDAD', kind='scatter')
plt.show()
# Con shift y tabulador podemos ver que otros tipos de graficos tiene este libreria
dataframe.plot(x='SALARIO', y='EDAD', kind='hist')
plt.show()

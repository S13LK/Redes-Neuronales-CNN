import numpy as np  # importamos la libreria de numpy

# primer ejemplo con un arreglo de numeros
numeros = {1, 2, 3, 4}
array = np.array("\nArreglo Simple \n\n" + str(numeros) + '\n')
print(array)

# genera numeros mediante rangos del zero al 30
primerG = np.arange(0, 30)
# genera numeros mediante rangos del zero al 30 de 4 en 4
segundoG = np.arange(0, 30, 4)
# genera numeros mediante zeros, en este caso serían 4 zeros
tercerG = np.zeros(4)
# genera numeros mediante unos, en este caso serían 10 unos
cuartoG = np.ones(10)
# genera numeros mediante unos, y lo hace en tres filas por dos columnas
quintoG = np.ones((3, 2))
# genera numeros aleatorios con linea space, desde el 0 al 30, y muestra 10 numeros
sextoG = np.linspace(0, 30, 10)
# genera numeros aleatorios del 0 al 100
septimoG = np.random.randint(0, 100)
# genera numeros aleatorios del 0 al 100, y muestra solo 10
octavoG = np.random.randint(0, 100, 10)
# obtenemos el maximo de un arreglo
novenoG = octavoG.max()
# obtenemos la posición del maximo de un arreglo
decimoG = octavoG.argmax()
# obtenemos el minimo de un arreglo
onceG = octavoG.min()
# obtenemos la posicion del minimo de un arreglo
doceG = octavoG.argmin()
# obtenemos la media del arreglo
treceG = octavoG.mean()
# genera 5 filas 2 columnas de mi arreglo de ejemplo
catorceG = octavoG.reshape(5, 2)
# obtenemos la fila 2 y la columna 0 del arreglo que generamos
# y así accedemos a los elementos del arreglo
quinceG = catorceG[2, 0]
# filtrar los elementos con el array que creamos
filtro = quinceG > 30
# En esta ultima variable le aplicamos el filtro que hemos creado
# aplicamos el filtro de un array para obtener un nuevo array con los datos filtrados
dieciseisG = catorceG[filtro]


print('\n' + str(primerG) + "\n")
print("\n" + str(segundoG) + '\n')
print('\n' + str(tercerG) + "\n")
print("\n" + str(cuartoG) + '\n')
print('\n' + str(quintoG) + "\n")
print("\n" + str(sextoG) + '\n')
print('\n' + str(septimoG) + "\n")
print("\n" + str(octavoG) + '\n')
print('\n' + str(novenoG) + "\n")
print("\n" + str(decimoG) + '\n')
print('\n' + str(onceG) + "\n")
print("\n" + str(doceG) + '\n')
print('\n' + str(treceG) + "\n")
print("\n" + str(catorceG) + '\n')
print('\n' + str(quinceG) + "\n")
print("\n" + str(dieciseisG) + '\n')

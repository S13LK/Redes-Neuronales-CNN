# Grafo por defecto o por default
import tensorflow as tf

# esta es la sintaxis del grafo por defecto
# pero marca error con la v2 de tensorflow
# tf.get_default_graph()
g = tf.import_graph_def
print('\n1er ejemplo corregido con otra sintaxis ' + str(g))
# este si funciona, pero con otra sintaxis
grafotrash = tf.Graph().as_default
print('\nbullshit ' + str(grafotrash))
# y así creamos un grafo con el metodo Graph()
grafo1 = tf.Graph()
print('\nok ' + str(grafo1))

# esto sirve para saber si el grafo 1 es nuestro grafo por defecto
# pero no funciona esta porqueria por la puta version 1 de TF
# with grafo1.as_default():
# esta linea sirve para saber si es el grafo por defecto
# pero como no usa el with nos devuelve falso
# en caso contrario sería verdadero
print(grafo1 is tf.import_graph_def)
# Aunque de esta manera si nos devuelve verdadero
# ya que en la variable g, guardamos un grafo por defecto
# formas de simplificar el uso de la versión 2 de TF
print(g is tf.import_graph_def)

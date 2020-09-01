# MNIST - Base de datos de imagenes de digitos escritos a mano
# 14/08/20
# Nos quedamos en esta secci+on, ya que hay que migrar a la v2 y utilizar keras
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras as k

# importamos el datasets de fashion_mnist con keras
# ya que keras es de alto nivel
fashion_mnist = k.datasets.fashion_mnist
# importamos las imagenes de entranamiento y las etiquetas de entrenamiento mediante keras
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()
# imprime el tipo de mnist
print('\n' + str(fashion_mnist) + '\n')

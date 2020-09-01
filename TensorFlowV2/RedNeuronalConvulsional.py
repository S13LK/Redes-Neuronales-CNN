# Red Neuronal Convulsional (CNN)
# 18/08/2020
# importamos las librerias
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# descraga y prepara el conjunto de datos CIFAR10
(train_images, train_labels), (test_images,
                               test_labels) = datasets.cifar10.load_data()
# normalizamos los valores de pixel entre 0 y 1
train_iamges, test_images = train_images / 255.0, test_images / 255.0
# verificamos los datos
class_names = ['airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# trazamos las primeras 25 imagenes del conjunto de datos de entrenamiento
# y mostramos el nombre de la clase debajo de cada imagen
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)  # cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# creamos la red neuronal convulcional
# utilizando un patron comun, una pila de capas CONV2 y MaxPooling2D
# una CNN toma tensores de forma (image_height, image_width, color_channels) ignorando el tamaño del lote
# color_channels se refiere a (R, G, B).
# configurará nuestra CNN para procesar entradas de forma (32, 32, 3), que es el formato de las imágenes CIFAR
# Puede hacer esto pasando el argumento input_shape a nuestra primera capa.
modelo = models.Sequential()
modelo.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
modelo.add(layers.MaxPooling2D((2, 2)))
modelo.add(layers.Conv2D(64, (3, 3), activation='relu'))
modelo.add(layers.MaxPooling2D((2, 2)))
modelo.add(layers.Conv2D(64, (3, 3), activation='relu'))

# muestra la arquitectura de nuestro modelo inicial
modelo.summary()

# agregamos capas densas en la parte superior 
# alimentará el último tensor de salida de la base convolucional
# (de forma (4, 4, 64)) en una o más capas densas para realizar la clasificación.
# las capas densas toman vectores como entrada (que son 1D)
# mientras que la salida actual es un tensor 3D. 
# Primero, aplanará (o desenrollará) la salida 3D a 1D
# luego agregará una o más capas densas en la parte superior. 
# CIFAR tiene 10 clases de salida, por lo que usa una capa Densa final con 10 salidas
# y una activación softmax.
modelo.add(layers.Flatten())
modelo.add(layers.Dense(64, activation='relu'))
modelo.add(layers.Dense(10))

# muestra la arquitetura completa de nuestro modelo
# como puede ver, nuestras salidas (4, 4, 64) se aplanaron en vectores de forma (1024)
# antes de pasar por dos capas densas.
modelo.summary()

# compilamos el modelo
modelo.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])
# entrenamos al modelo
history = modelo.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))
# evaluamos el modelo
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0.5, 1)
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = modelo.evaluate(test_images, test_labels, verbose=2)
print('\n' + str(test_acc) + '\n')

# Conclusiones
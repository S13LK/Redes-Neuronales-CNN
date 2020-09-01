# Uso Experto TF y Keras
# 18/08/2020
# importamos las librerias
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

# carga y prepara el conjunto de datos MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# agrega una dimensión de canales
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# utiliza tf.data para separar por lotes y mezclar el conjunto de daots
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(1000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(32)

# contruye el modelo, utilizando la API de Keras


class MiModelo(Model):
    def __init__(self):
        super(MiModelo, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(120, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


# creamos la instancia del modelo
modelo = MiModelo()
# función de perdida
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
# optimizador
optimizador = tf.keras.optimizers.Adam()
# metricas para medir la perdida y exactitud del modelo
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='test_accuracy')

# utiliza tf.GradientTape para entrenar el modelo


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predicciones = modelo(images)
        loss = loss_object(labels, predicciones)
    gradients = tape.gradient(loss, modelo.trainable_variables)
    optimizador.apply_gradients(zip(gradients, modelo.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predicciones)


@tf.function
def test_step(images, labels):
    predicciones = modelo(images)
    t_loss = loss_object(labels, predicciones)
    
    test_loss(t_loss)
    test_accuracy(labels, predicciones)


# probando el modelo
EPOCHS = 5

for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
    
    template = 'Epoch {}, Perdida: {}, Exactitud: {}, Perdida de prueba: {}, Exactitud de prueba: {}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          test_loss.result(),
                          test_accuracy.result()*100))
# reinicia las metricas para el siguiente epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

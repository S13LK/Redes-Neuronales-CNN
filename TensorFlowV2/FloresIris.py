# Ejemplo Flores de Iris
# 17/08/2020
# Importamos las librerias
import pandas as pd
import tensorflow as tf
import keras as k
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# carga el conjunto de datos
dataframe = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv', header=None)
dataset = dataframe.values
x = dataset[:,0,4].astype(float)
y = dataset[:,4]

# codifica la variable de salida
# codifica valores de la clase como enteros
encoder = LabelEncoder()
encoder.fit(y)
encoder_y = encoder.transform(y)
# convierte enteros para variables dummy
dummy_y = k.utils.np_utils.to_categorical(encoder_y)

# define el modelo de la red neuronal
def baseline_model():
    # crea el modelo
    model = k.Sequential()
    model.add(k.layers.Dense(8, input_dim=4, activation='relu'))
    model.add(k.layers.Dense(3, activation='softmax'))
    # compila el model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
    return model
    
estimator = k.wrappers.scikit_learn.KerasClassifier(build_fn=baseline_model, epochs=20, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
resultado = cross_val_score(estimator, x, dummy_y, cv=kfold)
print('Baseline: %.2f%%(%.2f%%)' % (resultado.mean()*100, resultado.std()*100))
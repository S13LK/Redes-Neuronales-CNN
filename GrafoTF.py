# Grafo en TensorFlow
# Son conjuntos de nosos, que estan conectados entre sí unos con otros
# Los conectores se llaman tensores

# reoresentación grafica de un Grafo con Tensorflow
# Nodo1(O)
#        > (O) Nodo3 ----> resultado
# Nodo2(O)

# importamos la libreria de tensorflow
import tensorflow as tf

with tf.compat.v1.Session() as session:

    # creamos la constante del primer nodo
    nodo1 = tf.constant(6)
    # creamos la constante del segundo nodo
    nodo2 = tf.constant(4)
    # creamos el tercer nodo, que es la sumatoria de los nodos 1 y 2
    nodo3 = nodo1 + nodo2
    # e imprimimos en pantalla el resultado del nodo 3
    # sin tener que estar utiliazando la maldita session
    # que como me caga el puto palo porque es de la version 1
    # y yo estoy ocupando la version 2
    #print('\n' + str(tf.print(nodo3)))

    # ahora lo veremos con esta pendejada de sesion

    resultado = session.run(nodo3)
    print(resultado)
    session.close()

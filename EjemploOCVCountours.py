# Countour Example with OpenCV
# Importamos las librerias
import numpy as np
from cv2 import cv2

# imagen de origen
im = cv2.imread('test.jpg')
# modo de recuperación del contorno
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# metodo de aproximación del contorno
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
# genera los contornos y la jerarquia
# contornos en una lista de python de todos los contornos de la imagen
# cada contorno individual es una matriz numpy de coordenadas (x,y) de los ountos de limite del objecto.
im2, contours, hierachy = cv2.findContours(
    thresh, cv2.RETR_TREE, cv2.CHAIN_APROX_SIMPLE)

# nota importante: se utiliza la función de cv2.drawContours
# también se puede utilizar para dibujar cualquier forma que tenga sus puntos de limite.
# su primer argumento es es la imagen de origen
# su segundo argumento son los contornos que se deben pasar como una lista de python
# su tercer argumento es indice de contronos(útil al dibujar contorno individual).
# Para dibujar todos los contornos, pase -1
# Los argumentos restantes son color, grosor, etc.

# Para dibujar todos los contornos de una imagen
#cv2.drawContours(img, contours, -1, (0,255,0), 3)
# Para dibujar un contorno individual, diga 4 contorno
#cv2.drawContours(img, contours, 3, (0,255,0), 3)
# Este metodo es utili, en la mayoria de la veces
#cnt = contours[4]
#cv2.drawContours(img, [cnt], 0, (0,255,0), 3)


# nota final pero muy importante sobre OpenCV y sus metodos
# cv2.CHAIM_APPROX_NONE, se almacena todos los puntos del limite
# En realidad necesitamos todos los puntos, por ejemplo:
# encontraste el contorno de una linea recta
# ¿Necesita todos los puntos de la linea para representar esa linea?
# No, solo necesitamos dos puntos finales de esa linea
# cv2.CHAIN_APPROX_SIMPLE, elimina todos los puntos redundantes y comprime el contorno.


# En conclusión el metodo simple, ahorra demaciada memoria, ya que no obtiene demaciados puntos innecesarios

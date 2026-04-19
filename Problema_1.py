# pip install opencv-python numpy matplotlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Ecualización local de histograma
def ecualizacion_local(imagen: np.ndarray, tamaño: int):
    '''
       Recibe dos cosas: la imagen y el tamaño de la ventana.
       Esa ventana es un cuadrado que va a ir recorriendo
       toda la imagen. 
    '''
    '''
       Se agregan bordes artificiales copiando los valores del borde real 
       así la ventana siempre puede existir completa.
    '''
    margen = tamaño // 2
    imagen_extendida = cv2.copyMakeBorder(imagen, margen, margen, margen, margen,
                                           cv2.BORDER_REPLICATE)
    filas, columnas = imagen.shape

    resultado = imagen.copy()

    for fila in range(filas):
        for columna in range(columnas):
            ventana = imagen_extendida[fila: fila + tamaño,
                                       columna: columna + tamaño]
            ventana_ecualizada = cv2.equalizeHist(ventana)
            resultado[fila, columna] = ventana_ecualizada[margen, margen]
    
    return resultado




'''Carga de la imagen y prueba con distintas ventanas'''
imagen = cv2.imread("Imagen_con_detalles_escondidos.tif", cv2.IMREAD_GRAYSCALE)
if imagen is None:
    raise FileNotFoundError("No se pudo cargar la imagen.")

base = plt.subplot(221)
plt.imshow(imagen, cmap="gray")
plt.title("Original")

tam = 7
salida = ecualizacion_local(imagen, tam)
plt.subplot(222, sharex=base, sharey=base)
plt.imshow(salida, cmap="gray")
plt.title(f"Ventana {tam}")

tam = 25
salida = ecualizacion_local(imagen, tam)
plt.subplot(223, sharex=base, sharey=base)
plt.imshow(salida, cmap="gray")
plt.title(f"Ventana {tam}")

tam = 45
salida = ecualizacion_local(imagen, tam)
plt.subplot(224, sharex=base, sharey=base)
plt.imshow(salida, cmap="gray")
plt.title(f"Ventana {tam}")

plt.show()
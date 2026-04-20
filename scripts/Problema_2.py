import cv2
import numpy as np
import matplotlib.pyplot as plt

def detectar_lineas(vector_pixeles: np.ndarray, umbral: int) -> list:
    """
    Detecta líneas a partir de la suma de píxeles por fila o columna.
    Devuelve una lista con los índices finales de cada línea detectada.
    """
    mascara_lineas = vector_pixeles > umbral
    indices = np.argwhere(mascara_lineas).flatten()

    if len(indices) == 0:
        return []
    finales = []
    ultimo = indices[0]

    for i in range(1, len(indices)):
        if indices[i] - indices[i - 1] > 1:
            finales.append(ultimo)
        ultimo = indices[i]
    finales.append(ultimo)
    return finales

def extraer_celdas(imagen_binaria: np.ndarray, columnas: list, filas: list) -> dict:
    """
    Recorta las celdas donde están las preguntas.
    Devuelve un diccionario {numero_pregunta: imagen_celda}.
    """
    celdas = {}
    numero_pregunta = 1

    for j in range(0, len(columnas) - 1, 2):
        for i in range(len(filas) - 1):
            y1, y2 = filas[i], filas[i + 1]
            x1, x2 = columnas[j], columnas[j + 1]
            recorte = imagen_binaria[y1:y2, x1:x2]
            celdas[numero_pregunta] = recorte
            numero_pregunta += 1
    return celdas

def extraer_zona_respuesta(celdas: dict) -> dict:
    """
    Busca dentro de cada celda la zona donde está la letra marcada.
    Devuelve {numero_pregunta: imagen_respuesta}.
    """
    zonas_respuesta = {}
    
    for numero, celda in celdas.items():
        cantidad, etiquetas, stats, centroides = cv2.connectedComponentsWithStats(celda, 8, cv2.CV_32S)
        for stat in stats:
            ancho = stat[2]
            if 50 < ancho < 150:
                x = stat[0]
                y = stat[1]
                w = stat[2]
                zonas_respuesta[numero] = celda[y - 13:y - 2, x:x + w]
    return zonas_respuesta

def reconocer_letra(zonas_respuesta: dict) -> dict:
    """
    Intenta reconocer si la marca encontrada es A, B, C o D.
    Si no puede identificarla bien, devuelve MAL.
    """
    letras_detectadas = {}

    for numero, zona in zonas_respuesta.items():
        cantidad, etiquetas, stats, centroides = cv2.connectedComponentsWithStats(zona, 8, cv2.CV_32S)
        letra = "MAL"

        if cantidad == 2:
            contornos, _ = cv2.findContours(zona, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cantidad_contornos = len(contornos)

            if cantidad_contornos == 1:
                letra = "C"
            elif cantidad_contornos == 3:
                letra = "B"
            elif cantidad_contornos == 2:
                area = cv2.contourArea(contornos[0])
                if area == 23.0:
                    letra = "A"
                else:
                    letra = "D"
        letras_detectadas[numero] = letra
    return letras_detectadas

def corregir_examen(imagen_binaria: np.ndarray, columnas: list, filas: list) -> int:
    solucion = {
        1: "C", 2: "B", 3: "A", 4: "D", 5: "B",
        6: "B", 7: "A", 8: "B", 9: "D", 10: "D"
    }

    celdas = extraer_celdas(imagen_binaria, columnas, filas)
    zonas = extraer_zona_respuesta(celdas)
    letras = reconocer_letra(zonas)
    puntaje = 0

    for numero in range(1, 11):
        letra = letras.get(numero, "MAL")

        if letra == solucion[numero]:
            puntaje += 1
            print(f"Pregunta {numero}: OK")
        else:
            print(f"Pregunta {numero}: MAL")
    return puntaje

def extraer_campos_encabezado(imagen_binaria: np.ndarray, filas: list) -> list:
    """
    Recorta el encabezado y extrae los campos Name, Date y Class.
    Devuelve una lista con esas subimágenes.
    """
    y1, y2 = 0, filas[0] - 2
    x1, x2 = 0, imagen_binaria.shape[1]

    encabezado = imagen_binaria[y1:y2, x1:x2]

    campos = []
    cantidad, etiquetas, stats, centroides = cv2.connectedComponentsWithStats(encabezado, 8, cv2.CV_32S)

    for stat in stats:
        ancho = stat[2]
        if 50 < ancho < 200:
            x = stat[0]
            y = stat[1]
            w = stat[2]
            campos.append(encabezado[y - 20:y - 2, x:x + w])
    return campos

def contar_palabras_y_caracteres(campo: np.ndarray) -> tuple[int, int]:
    """
    Cuenta palabras y caracteres aproximando cada carácter como componente conectada.
    """
    cantidad, etiquetas, stats, centroides = cv2.connectedComponentsWithStats(campo, 8, cv2.CV_32S)

    caracteres = 0
    palabras = 0
    x_anterior = None

    for i in range(1, cantidad):
        x, y, w, h, area = stats[i]
        caracteres += 1

        if x_anterior is not None:
            distancia = x - x_anterior
            if distancia > 15:
                palabras += 1
        x_anterior = x
    if caracteres > 0:
        palabras += 1
    return palabras, caracteres

def validar_datos_encabezado(imagen_binaria: np.ndarray, filas: list) -> np.ndarray:
    """
    Valida Name, Date y Class.
    Devuelve la imagen del campo Name umbralizada/invertida para usar después.
    """
    campos = extraer_campos_encabezado(imagen_binaria, filas)

    cantidad_palabras, cantidad_caracteres = contar_palabras_y_caracteres(campos[0])
    if cantidad_palabras >= 2 and cantidad_caracteres <= 25:
        print("Name: OK")
    else:
        print("Name: MAL")

    cantidad_palabras, cantidad_caracteres = contar_palabras_y_caracteres(campos[1])
    if cantidad_palabras == 1 and cantidad_caracteres == 8:
        print("Date: OK")
    else:
        print("Date: MAL")

    _, cantidad_caracteres = contar_palabras_y_caracteres(campos[2])
    if cantidad_caracteres == 1:
        print("Class: OK")
    else:
        print("Class: MAL")

    _, nombre_binario = cv2.threshold(campos[0], 0, 255, cv2.THRESH_BINARY_INV)
    return nombre_binario

archivos_examen = {
    1: "../assets/examen_1.png",
    2: "../assets/examen_2.png",
    3: "../assets/examen_3.png",
    4: "../assets/examen_4.png",
    5: "../assets/examen_5.png"
}

condiciones = {}
imagenes_nombres = {}

for numero_examen, ruta in archivos_examen.items():
    imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)

    if imagen is None:
        raise FileNotFoundError(f"No se pudo cargar {ruta}")

    _, imagen_binaria = cv2.threshold(imagen, 190, 1, cv2.THRESH_BINARY_INV)

    suma_columnas = np.sum(imagen_binaria, axis=0)
    suma_filas = np.sum(imagen_binaria, axis=1)

    columnas_detectadas = detectar_lineas(suma_columnas, 600)
    filas_detectadas = detectar_lineas(suma_filas, 450)

    print(f"\nExamen {numero_examen}:")

    imagenes_nombres[numero_examen] = validar_datos_encabezado(imagen_binaria, filas_detectadas)

    nota = corregir_examen(imagen_binaria, columnas_detectadas, filas_detectadas)

    print(f"Calificación: {nota}")

    if nota >= 6:
        estado = "APROBADO"
    else:
        estado = "DESAPROBADO"

    print(estado)
    condiciones[numero_examen] = estado

# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------

panel = np.ones((700, 700, 3), dtype=np.uint8) * 255

verde = (0, 180, 0)
rojo = (0, 0, 220)
negro = (0, 0, 0)

fuente = cv2.FONT_HERSHEY_SIMPLEX

# Titulo
cv2.putText(panel, "RESULTADOS EXAMENES", (170, 40), fuente, 0.8, negro, 2, cv2.LINE_AA)

# Subtitulos
cv2.putText(panel, "APROBADOS", (70, 90), fuente, 0.7, verde, 2, cv2.LINE_AA)

cv2.putText(panel, "DESAPROBADOS", (380, 90), fuente, 0.7, rojo, 2, cv2.LINE_AA)

# Posiciones iniciales
y_aprobados = 120
y_desaprobados = 120

for numero_examen, estado in condiciones.items():
    nombre_img = imagenes_nombres[numero_examen]
    nombre_img = cv2.cvtColor(nombre_img, cv2.COLOR_GRAY2BGR)

    alto, ancho = nombre_img.shape[:2]

    if estado == "APROBADO":
        panel[y_aprobados:y_aprobados + alto, 40:40 + ancho] = nombre_img
        cv2.rectangle(
            panel,
            (40, y_aprobados),
            (40 + ancho, y_aprobados + alto),
            verde,
            2
        )
        y_aprobados += alto + 15
    else:
        panel[y_desaprobados:y_desaprobados + alto, 380:380 + ancho] = nombre_img
        cv2.rectangle(
            panel,
            (380, y_desaprobados),
            (380 + ancho, y_desaprobados + alto),
            rojo,
            2
        )
        y_desaprobados += alto + 15

plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
plt.title("Resultados finales")
plt.axis("off")
plt.show()
cv2.imwrite("../assets/resultados_finales.png", panel)
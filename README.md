# Procesamiento de Imagenes I
## TRABAJO PRÁCTICO N° 1 - Año 2026 - 1° Semestre
### Entorno Virtual
Para instalar las dependencias es necesario crear un entorno virtual para aislar las dependencias del proyecto. Para ello, ejecuta el siguiente comando en la terminal:

1. **Creación del entorno:**
   ```bash
   python -m venv .venv
   ```

2. **Activación del entorno:**
   - En Windows:
     ```powershell
     .\.venv\Scripts\activate
     ```
   - En Linux o macOS:
     ```bash
     source .venv/bin/activate
     ```

### Instalación de dependencias
Una vez activado el entorno virtual, es necesario instalar las librerías del proyecto:
- **OpenCV** (`opencv-python`): Principal librería para procesamiento y manipulación de imágenes.
- **NumPy** (`numpy`): Para el manejo eficiente de los arreglos y matrices que representan a las imágenes.
- **Matplotlib** (`matplotlib`): Utilizada para graficar y mostrar resultados de imágenes con histogramas.

Para instalarlas, ejecuta el siguiente comando en la terminal:
```bash
pip install -r requirements.txt
```

### Ejecución de scripts
Para ejecutar cada uno de ellos y ver sus resultados, ejecuta los siguientes comandos en la terminal:

**Problema 1:**
```bash
cd ./scripts
python Problema_1.py
```

**Problema 2:**
```bash
cd ./scripts
python Problema_2.py
```
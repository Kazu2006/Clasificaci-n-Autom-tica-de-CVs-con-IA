
# 🧠 Clasificador Inteligente de CVs en PDF con IA

Este proyecto usa inteligencia artificial para analizar y clasificar automáticamente currículums vitae (CVs) en formato PDF, agrupándolos en tres categorías: **Alto**, **Medio** o **Bajo**. Es ideal para procesos de selección que necesitan filtrar rápidamente candidatos según su perfil profesional.

---

## 🚀 ¿Cómo probar este proyecto desde cero?

### 1. Crea y activa un entorno virtual (opcional pero recomendado)

```bash
# Windows PoweShell como administrador
     python -m venv venv
```

---

### 2. Instala los paquetes necesarios

```bash
pip install -r requirements.txt
```

---

### 3. Prepara los datos (¡muy importante!)

1. Crea una carpeta `data/ejemplos/` si no existe.
2. Copia dentro **tus CVs en PDF** (ejemplo: `cv01.pdf`, `cv02.pdf`, etc.).
3. Crea un archivo llamado `data/cvs_etiquetados.csv` con este formato:

```csv
ruta,texto,etiqueta
data/ejemplos/cv01.pdf,,Alto
data/ejemplos/cv02.pdf,,Medio
data/ejemplos/cv03.pdf,,Bajo
```

> ⚠️ La columna `texto` debe dejarse vacía. El sistema la rellenará automáticamente al entrenar.

---

### 4. Entrena el modelo

```bash
python train_predict.py --train
```

Esto:

* Extraerá el texto real de los PDFs.
* Entrenará un modelo usando ese texto.
* Guardará el modelo en `cv_pipeline.joblib`.

---

### 5. Prueba la predicción de un nuevo CV 

```bash
python train_predict.py --predict data/ejemplos/cv01.pdf
```

Verás algo así:

```txt
Probabilidades:
 - Alto: 42.5%
 - Medio: 22.1%
 - Bajo: 5.4%

Predicción base: Alto
Decisión automática: Aprobado y pasa al siguiente filtro
```

---

### 🧪 ¿Qué hace internamente?

* Extrae texto desde el PDF automáticamente.
* Lo convierte en vectores usando **TF-IDF**.
* Usa **Logistic Regression** para predecir la categoría del perfil.
* Aplica reglas de negocio para decidir si el CV se aprueba, rechaza o revisa.

---

### 📦 ¿Y si quiero probar más rápido?

Puedes usar los ejemplos de CVs proporcionados, o generar los tuyos en Word y exportarlos como PDF. El sistema no depende de nombres específicos, solo del contenido.

---

### ✅ Requisitos

* Python 3.8 o superior
* pip
* PDF legibles (no escaneados)

---

### ✨ ¿Ideas para mejorar?

* Agregar interfaz gráfica con Streamlit o Flask
* Mostrar palabras más influyentes por clase
* Exportar el resultado en JSON o Excel

---

### 🧩 Teconologias Utilizadas
* Python (base del sistema)
* pdfplumber (para extraer texto desde PDFs)
* scikit-learn (modelo de IA y procesamiento de texto)
* TfidfVectorizer (representación de texto)
* LogisticRegression o SVC (modelos de clasificación)
* argparse (para línea de comandos)
* pandas (manejo del CSV)

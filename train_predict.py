import argparse
import os
import pdfplumber
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

# Rutas fijas
MODEL_PATH = 'cv_pipeline.joblib'
CSV_PATH   = 'data/cvs_etiquetados.csv'

# 1) Extraer texto de PDF
def extract_text_from_pdf(path):
    text = ''
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += ' ' + (page.extract_text() or '')
    return text.strip()

# 2) Entrenamiento del modelo
def train():
    df = pd.read_csv(CSV_PATH)
    # Si aún no hay texto extraído, lo llenamos
    if 'texto' not in df.columns or df['texto'].isnull().all() or df['texto'].str.len().sum() == 0:
        df['texto'] = df['ruta'].apply(extract_text_from_pdf)
        df.to_csv(CSV_PATH, index=False)

    X = df['texto']
    y = df['etiqueta']
    pipeline = make_pipeline(
        TfidfVectorizer(max_features=5000, ngram_range=(1,2)),
        LogisticRegression(solver='liblinear', multi_class='ovr')
    )
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Modelo entrenado y guardado en: {MODEL_PATH}")

# 3) Predicción con reglas de umbral
def predict(pdf_path):
    if not os.path.exists(MODEL_PATH):
        print("Error: modelo no encontrado. Ejecuta primero con --train.")
        return
    pipeline = joblib.load(MODEL_PATH)

    # Extraer y procesar texto
    text = extract_text_from_pdf(pdf_path)
    probs = pipeline.predict_proba([text])[0]
    labels = pipeline.classes_

    # Mostrar probabilidades
    print("Probabilidades:")
    for lbl, p in zip(labels, probs):
        print(f" - {lbl}: {p*100:.1f}%")

    # Predicción base (clase con mayor probabilidad)
    base_pred = labels[probs.argmax()]
    print(f"\nPredicción base: {base_pred}")

    # Umbrales demo ajustados a 30%
    UMBRAL_ALTO = 0.30
    UMBRAL_BAJO = 0.30

    if 'Alto' in labels and probs[list(labels).index('Alto')] >= UMBRAL_ALTO:
        print("Decisión automática: Aprobado y pasa al siguiente filtro")
    elif 'Bajo' in labels and probs[list(labels).index('Bajo')] >= UMBRAL_BAJO:
        print("Decisión automática: Rechazado automáticamente")
    else:
        print("Decisión automática: Revisión manual")

# 4) CLI
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrena o predice CVs con umbrales')
    parser.add_argument('--train', action='store_true', help='Entrenar modelo')
    parser.add_argument('--predict', type=str, help='Ruta al PDF a predecir')
    args = parser.parse_args()
    if args.train:
        train()
    elif args.predict:
        predict(args.predict)
    else:
        parser.print_help()

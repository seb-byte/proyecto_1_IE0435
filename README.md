# Detección de Contaminación en Imágenes — SVM

Clasificación binaria de imágenes 128×128 píxeles para detectar contaminación (granos de arroz u objetos extraños) en superficies. El pipeline binariza las imágenes y entrena un clasificador SVM con kernel RBF sobre los 16 384 valores de píxel resultantes.

---

## Instalación

```bash
pip install -r requirements.txt
```

---

## Estructura del proyecto

```
proyecto_1_IE0435/
│
├── src/                            # Código fuente
│   ├── models.py                   # Clases ThresholdedRF, ThresholdedSVM, híbridos
│   ├── generate_dataset.py         # Convierte imágenes a CSV
│   ├── train.py                    # Entrena el modelo SVM
│   ├── predict.py                  # Inferencia desde carpeta de imágenes
│   └── evaluate.py                 # Evaluación sobre CSV con métricas
│
├── data/
│   ├── raw/
│   │   ├── positive/               # ← Colocar aquí las fotos CON contaminación
│   │   └── negative/               # ← Colocar aquí las fotos SIN contaminación
│   └── processed/
│       ├── train.csv               # Dataset de entrenamiento (200 filas, generado)
│       └── test.csv                # Dataset de prueba (60 filas, generado)
│
├── models/
│   └── c26797_sebastian_rojas.joblib   # Modelo SVM ya entrenado
│
├── reports/
│   ├── figures/                    # Imágenes de verificación de binarización (generadas)
│   └── Proyecto_1_Informe_Final_C26797.pdf
│
├── DATASET.md                      # Descripción del dataset y proceso de recolección
├── MODEL_CARD.md                   # Ficha del modelo (métricas, limitaciones, ética)
├── README.md
├── LICENSE
└── requirements.txt
```

> **Nota:** Las carpetas `data/raw/positive/` y `data/raw/negative/` están en `.gitignore` — las imágenes no se suben al repositorio.

---

## Flujo de trabajo

### Paso 1 — Colocar las fotos

Copiar las imágenes en las carpetas correspondientes:

```
data/raw/positive/   ← fotos CON contaminación (grano de arroz, objeto extraño)
data/raw/negative/   ← fotos SIN contaminación (superficie limpia)
```

Formatos soportados: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`

---

### Paso 2 — Generar el dataset

```bash
python src/generate_dataset.py
```

Esto procesa todas las imágenes en `data/raw/`, las binariza y genera:

- `data/processed/train.csv` — set de entrenamiento
- `data/processed/test.csv` — set de prueba
- `data/processed/dataset.csv` — dataset completo
- `reports/figures/` — PNGs de verificación de la binarización

Ajustar en el encabezado de `src/generate_dataset.py` si es necesario:

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `TRAIN_POR_CLASE` | 100 | Muestras por clase en train |
| `TEST_POR_CLASE` | 30 | Muestras por clase en test |

---

### Paso 3 — Entrenar el modelo

```bash
python src/train.py
```

El script:
1. Carga `data/processed/train.csv` y separa en 80% train / 20% validación.
2. Entrena `StandardScaler → PCA(30) → SVC(RBF, C=20, γ=0.003)`.
3. Busca el umbral óptimo maximizando `geometric_recall` en validación.
4. Imprime accuracy, recall+, recall−, F1 y ROC-AUC.
5. Guarda el modelo en `models/svm_model.joblib`.

**Parámetros clave** (editables en `src/train.py`):

| Parámetro | Valor | Descripción |
|---|---|---|
| `N_COMPONENTS` | 15 | Componentes PCA |
| `C` | 36.355 | Regularización SVM |
| `GAMMA` | 0.00221 | Ancho del kernel RBF |
| `TARGET_RECALL` | 0.90 | Objetivo de recall mínimo por clase |

---

### Paso 4a — Inferencia desde carpeta de imágenes

```bash
python src/predict.py <ruta_carpeta>
```

**Modo predicción** — la carpeta contiene imágenes directamente (sin etiquetas):

```
fotos_nuevas/
  foto1.jpg
  foto2.jpg
```

```bash
python src/predict.py fotos_nuevas/
```

**Modo evaluación** — la carpeta contiene subcarpetas `positive/` y `negative/`:

```
fotos_prueba/
  positive/
    img_a.jpg
  negative/
    img_b.jpg
```

```bash
python src/predict.py fotos_prueba/
```

El modo se detecta automáticamente. En modo evaluación se calculan accuracy, precision, recall y F1.
El resultado se guarda como `predicciones_SVM_<timestamp>.csv` en la raíz del proyecto.

---

### Paso 4b — Evaluación sobre CSV

```bash
python src/evaluate.py data/processed/test.csv
```

O en modo interactivo (el script pedirá la ruta y opciones):

```bash
python src/evaluate.py
```

El script genera:
- Reporte en consola con accuracy, precision, recall+, recall−, F1 y matriz de confusión.
- `test3_resultados_predicciones_<timestamp>.csv` — predicciones por muestra.
- `test3_resultados_metricas_<timestamp>.csv` — tabla resumen de métricas.
- Gráficas de matrices de confusión (si hay etiquetas).

---

## Parámetros de preprocesado

Los valores en `src/predict.py` **deben coincidir** con los usados al generar el dataset en `src/generate_dataset.py`:

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `IMG_SIZE` | 128 | Tamaño de imagen en píxeles |
| `USAR_UMBRAL_ADAPT` | `True` | Umbral adaptativo Gaussiano |
| `ADAPT_BLOCK` | 31 | Tamaño del bloque local |
| `ADAPT_C` | 5 | Constante sustraída al umbral |
| `USAR_CLAHE` | `False` | Ecualización de histograma local |
| `USAR_GAMMA` | `False` | Corrección de gamma |

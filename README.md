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
├── generar_dataset.py      # Convierte carpetas de imágenes a CSV
├── train.csv               # Dataset de entrenamiento (200 filas)
├── test.csv                # Dataset de prueba (60 filas)
│
├── fotos/
│   └── negative/           # Fotos Negativas
│   └── positive/           # Fotos Positivas
│
├── svm.py                  # Entrena el modelo SVM
│
├── c26797_sebastian_rojas.joblib # Modelo ya entrenado
│
├── model_wrappers.py       # Clase ThresholdedRF (wrapper pipeline + umbral)
├── predict_folder.py       # Inferencia directa desde carpeta de imágenes
│
├── README.md               # Este archivo — instrucciones de uso
├── DATASET.md              # Descripción del dataset y proceso de recolección
├── MODEL_CARD.md           # Ficha del modelo (métricas, limitaciones, ética)
├── LICENSE                 # Licencia MIT
├── requirements.txt        # Dependencias Python
│
└── reports/
    └── informe_final.md    # Informe técnico completo
```

---

## 1. Preparar el dataset

Colocar las imágenes en dos carpetas:

```
positive/   <- imágenes con contaminación  (etiqueta 1)
negative/   <- imágenes sin contaminación  (etiqueta 0)
```

Luego ejecutar:

```bash
python generar_dataset.py
```

Esto genera `dataset.csv`, `train.csv` y `test.csv`.
Ajustar `TRAIN_POR_CLASE` y `TEST_POR_CLASE` en el encabezado del script según el número de imágenes disponibles.

---

## 2. Entrenamiento

```bash
python SVM/svm.py
```

El script:
1. Carga `train.csv` y divide en 80% entrenamiento / 20% validación.
2. Entrena un pipeline `StandardScaler -> PCA(30) -> SVC(RBF, C=20, gamma=0.003)`.
3. Busca el umbral óptimo en el split de validación maximizando `geometric_recall`.
4. Imprime métricas finales (accuracy, recall+, recall-, F1, ROC-AUC).
5. Guarda el modelo en `svm_model.joblib`.

**Parámetros clave** (editables en el encabezado de `SVM/svm.py`):

| Parámetro | Valor | Descripción |
|---|---|---|
| `N_COMPONENTS` | 30 | Componentes PCA |
| `C` | 20.0 | Regularización SVM |
| `GAMMA` | 0.003 | Ancho del kernel RBF |
| `TARGET_RECALL` | 0.90 | Objetivo de recall para ambas clases |

---

## 3. Inferencia desde carpeta de imágenes
La idea de esta prueba es simular el ambiente real, por lo que se encarga de recibir fotos como input, pasarlas a la matriz de ceros y unos, pasarle estos datos al modelo y hacer la predicción. Esta prueba cuenta con dos modos:

### Modo predicción (sin etiquetas reales)

```
carpeta_fotos/
  imagen1.jpg
  imagen2.jpg
```

```bash
python predict_folder.py carpeta_fotos/
```

### Modo evaluación (con etiquetas implícitas en subcarpetas)

```
carpeta_fotos/
  positive/
    img_a.jpg
  negative/
    img_b.jpg
```

```bash
python predict_folder.py carpeta_fotos/
```

El modo se detecta automáticamente. En modo evaluación se calculan accuracy, precision, recall y F1.
El resultado se guarda en:

```
predicciones_SVM_<timestamp>.csv
```

Columnas modo predicción: `imagen, prediccion, etiqueta, probabilidad`
Columnas modo evaluación: `imagen, etiqueta_real, real_label, prediccion, pred_label, probabilidad`

---

## 4. Prueba básica sobre CSV (test.py)

Una vez entrenado el modelo, se puede evaluar directamente sobre un CSV generado con `generar_dataset.py`:

```bash
python test.py
```

El script pedirá:
1. **Ruta del CSV** — por defecto usa `dataset_test.csv`. Puede indicarse `test.csv` para evaluar sobre datos nunca vistos durante el entrenamiento.
2. **Si el CSV incluye etiquetas** — responder `1` (Sí) para calcular métricas, `2` (No) para solo obtener predicciones.
3. **Qué modelo probar** — seleccionar `4` para el SVM, o `A` para comparar todos los modelos disponibles.

El script genera:
- Reporte en consola con accuracy, precision, recall+, recall−, F1 y matriz de confusión.
- `resultados_predicciones_<timestamp>.csv` — predicciones por muestra.
- `resultados_metricas_<timestamp>.csv` — tabla resumen de métricas.
- Gráficas de matrices de confusión y comparación de métricas (si hay etiquetas).

También puede pasarse el CSV directamente como argumento (asume que tiene etiquetas):

```bash
python test.py test.csv
```

---

## Parámetros de preprocesado

Los parámetros de `predict_folder.py` deben coincidir con los usados en `generar_dataset.py`:

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `IMG_SIZE` | 128 | Tamaño de imagen en píxeles |
| `USAR_UMBRAL_ADAPT` | `True` | Umbral adaptativo Gaussiano |
| `ADAPT_BLOCK` | 31 | Tamaño del bloque local |
| `ADAPT_C` | 5 | Constante sustraída al umbral |
| `USAR_CLAHE` | `False` | Ecualización de histograma local |
| `USAR_GAMMA` | `False` | Corrección de gamma |

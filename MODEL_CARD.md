# Model Card — Detección de Contaminación en Imágenes

**Modelo:** SVM RBF con umbral optimizado (`ThresholdedRF` wrapper)
**Archivo:** `c26797_sebastian_rojas.joblib`
**Versión:** 15.2
**Fecha:** 2026-05

---

## Nombre del modelo

**SVM RBF con umbral optimizado** (`c26797_sebastian_rojas.joblib`)
Wrapper: `ThresholdedRF(pipeline, threshold=0.515)` — embebe el umbral dentro del mismo objeto serializado, sin archivo separado.
Pipeline interno: `StandardScaler → PCA(n_components=15) → SVC(kernel=rbf, C=36.355, γ=0.00221, class_weight=balanced)`.

---

## Uso previsto

**Aplicación objetivo:** detección automática de contaminación (granos de arroz u objetos extraños) en procesos de manofactura.

**Usuarios previstos:** sistemas de inspección de calidad en entornos de producción donde se disponga de cámara fija y fondo predominantemente blanco.

**Fuera de alcance:**
- Imágenes con fondo texturado, oscuro o de color.
- Contaminantes distintos a granos de arroz (no representados en el entrenamiento).
- Imágenes con oclusión parcial severa o múltiples objetos superpuestos.
- Video en tiempo real sin previa validación del pipeline de captura.

---

## Resumen del dataset

El dataset fue recolectado a partir del aporte de fotos de múltiples compañeros. Todas estas fotos se descargaron y se pre-procesaron del mismo modo para tener datos congruentes.

| Característica | Valor |
|---|---|
| Tipo de entrada | Imagen binaria 128×128 aplanada (16 384 features) |
| Clases | 0 = Negativo (limpio), 1 = Positivo (contaminado) |
| Train | 100 positivos + 100 negativos = 200 muestras |
| Test | 30 positivos + 30 negativos = 60 muestras |
| Balance | 50/50 artificial |
| Preprocesado | Redimensión → escala de grises → blur gaussiano → umbral adaptativo → vector 0/1 |

**Variaciones de captura no controladas:** iluminación entre sesiones, dispositivo de cámara, distancia y ángulo de captura. El umbral adaptativo mitiga parcialmente la variación de iluminación dentro de una imagen, pero no entre cámaras o condiciones de luz distintas.

---

## Proceso de etiquetado

- **Herramienta:** organización manual en carpetas `positive/` y `negative/`.
- **Criterio:** presencia visible de al menos un grano de arroz → etiqueta 1; superficie completamente limpia → etiqueta 0.
- **Revisión:** inspección visual directa por el operador del proyecto.
- **Limitaciones:** no se realizó doble revisión ni validación cruzada entre anotadores. Casos ambiguos (grano parcialmente visible, imagen desenfocada) pueden introducir ruido en las etiquetas.

---

## Métricas

El optimizador de umbral es **geometric recall** = √(recall_neg × recall_pos), que penaliza sesgo hacia cualquiera de las dos clases. El umbral resultante es **0.515**.

### Glosario de métricas

Las siguientes métricas se obtienen a partir de la matriz de confusión (TP = verdaderos positivos, TN = verdaderos negativos, FP = falsos positivos, FN = falsos negativos).

| Métrica | Fórmula | Qué mide |
|---|---|---|
| **Accuracy** | (TP + TN) / total | Fracción de predicciones correctas sobre todas las muestras. Engañosa en datasets desbalanceados. |
| **Balanced Accuracy** | (Recall₊ + Recall₋) / 2 | Promedio de recall por clase; insensible al desbalance. Equivale a accuracy cuando el dataset es 50/50. |
| **Precision** (clase positiva) | TP / (TP + FP) | De todo lo que el modelo predijo como contaminado, ¿qué fracción realmente lo estaba? Penaliza las falsas alarmas. |
| **Recall** (Recall positivo / sensibilidad) | TP / (TP + FN) | De todos los casos realmente contaminados, ¿qué fracción detectó el modelo? Penaliza los contaminantes no detectados. |
| **Recall negativo** (especificidad) | TN / (TN + FP) | De todos los casos realmente limpios, ¿qué fracción clasificó correctamente? Penaliza clasificar como contaminado algo limpio. |
| **F1-Score** | 2 · (Precision · Recall) / (Precision + Recall) | Media armónica de precision y recall. Útil cuando importa tanto no perder positivos como no generar falsas alarmas. |
| **ROC-AUC** | Área bajo la curva ROC | Probabilidad de que el modelo asigne mayor score a un positivo aleatorio que a un negativo aleatorio. 0.5 = azar, 1.0 = perfecto. Independiente del umbral. |
| **Geometric Recall** | √(Recall₋ × Recall₊) | Métrica de optimización del umbral. Cero si cualquiera de los dos recalls es 0; se maximiza solo cuando ambos son altos simultáneamente. |

> **¿Por qué priorizar recall sobre accuracy?** En inspección de calidad, un falso negativo (contaminante no detectado) tiene mayor costo que un falso positivo (producto limpio rechazado), por lo que se optimiza recall sobre precision. El geometric recall obliga a equilibrar la detección de ambas clases.

### Conjunto de entrenamiento (`train.csv` — evaluación con `src/evaluate.py`)

200 muestras (100 positivos + 100 negativos). Evalúa el ajuste del modelo sobre datos conocidos.

| Métrica | Negativo | Positivo | Global |
|---|---|---|---|
| Precision | 0.92 | 0.88 | — |
| Recall | 0.88 | 0.91 | — |
| F1-Score | 0.90 | 0.90 | — |
| Accuracy | — | — | **0.900** |
| Balanced Accuracy | — | — | 0.900 |

**Matriz de confusión (train):**

|  | Pred. Negativo | Pred. Positivo |
|---|---|---|
| **Real Negativo** | 88 (TN) | 12 (FP) |
| **Real Positivo** | 9 (FN) | 91 (TP) |

---

### Conjunto de prueba final (`test.csv` — evaluación independiente)

60 muestras (30 positivos + 30 negativos).

| Métrica | Negativo | Positivo | Global |
|---|---|---|---|
| Precision | 0.77 | 0.71 | — |
| Recall | 0.67 | 0.80 | — |
| F1-Score | 0.71 | 0.75 | — |
| Accuracy | — | — | **0.75** |
| Balanced Accuracy | — | — | 0.75 |


**Matriz de confusión (test):**

|  | Pred. Negativo | Pred. Positivo |
|---|---|---|
| **Real Negativo** | 20 (TN) | 10 (FP) |
| **Real Positivo** | 6 (FN) | 24 (TP) |

---

### Evaluación vs. objetivo declarado

**Objetivo:** ≥ 90% recall en ambas clases.

| Clase | Recall (train) | Recall (test) | ¿Objetivo alcanzado? |
|---|---|---|---|
| Negativo (limpio) | 0.88 | 0.67 | No (en test) |
| Positivo (contaminado) | 0.91 | 0.80 | No (en test) |

> El modelo alcanza el 91% y 88% de recall sobre los datos de entrenamiento, pero cae a 80% y 67% en el conjunto de prueba independiente. El objetivo de ≥90% en ambas clases no se cumple en condiciones de generalización, lo cual es consistente con el tamaño reducido del dataset (200 muestras) y la variabilidad de captura no controlada.

---

## Notas éticas y de seguridad

- **Sesgo por iluminación:** el modelo entrenado bajo condiciones de luz específicas puede degradarse ante cambios no contemplados.
- **Sesgo por fondo:** el pipeline asume fondo blanco; superficies de otro color pueden producir tasas de falsos positivos muy altas.
- **Sesgo por cámara:** diferentes sensores producen distribuciones de píxeles distintas. Un modelo entrenado con una cámara puede fallar con otra sin reentrenamiento.
- **Confianza mal calibrada:** las probabilidades de salida (Platt scaling) no están calibradas de forma independiente; no deben interpretarse como probabilidades bayesianas.
- **Uso crítico:** el modelo no es apto como único mecanismo de detección en aplicaciones donde una contaminación no detectada tenga consecuencias graves.

---

## Limitaciones técnicas

- **Objetos pequeños:** granos que ocupen menos del ~2% de la imagen (≈ 5×5 px tras redimensionar) pueden no producir suficiente contraste tras la binarización.
- **Oclusión:** granos cubiertos por otros objetos no son detectables en imágenes planas 2D.
- **Desenfoque:** imágenes borrosas producen binarizaciones con artefactos que degradan la predicción.
- **Dataset pequeño:** 200 muestras de entrenamiento producen alta varianza.
- **PCA fijo:** n_components=30 fue seleccionado por búsqueda de hiperparámetros sobre train.csv; puede no ser óptimo para nuevas distribuciones de datos.

---

## Reproducibilidad

### Requisitos

```bash
pip install -r requirements.txt
# Python 3.10+, Windows 11
```

### Entrenamiento

```bash
python src/train.py
# Parámetros fijos: N_COMPONENTS=15, C=36.35502692537726, GAMMA=0.0022112559088427667
# random_state=42 en PCA, SVC y train_test_split
```

### Inferencia

```bash
# Desde carpeta de imágenes (seleccionar modelo 4 = SVM en el menú)
python predict_folder.py ruta/a/mis/fotos/
```

### Hardware usado

- Plataforma: Windows 11
- CPU: un solo núcleo (el entrenamiento SVM no usa n_jobs)
- GPU: no requerida
- RAM: < 1 GB para el dataset de 200 muestras

### Semillas fijas

| Componente | Semilla |
|---|---|
| `train_test_split` | 42 |
| `PCA` | 42 |
| `SVC` | 42 |
| Dataset split (`generar_dataset.py`) | 42 |

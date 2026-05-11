# Documentación del Dataset

## Descripción general

El dataset consiste en imágenes binarias de superficies, capturadas para detectar la presencia de granos de arroz u objetos extraños (contaminación).

| Partición | Positivos | Negativos | Total |
|---|---|---|---|
| `train.csv` | 100 | 100 | 200 |
| `test.csv` | 30 | 30 | 60 |

Cada fila del CSV contiene 16 384 valores enteros (0 o 1) seguidos de la etiqueta (0 = negativo, 1 = positivo).

---

## Recolección

Las imágenes capturadas y las de los demás compañeros se organizaron manualmente en dos carpetas:

- `positive/` — imágenes que contienen contaminación (etiqueta 1)
- `negative/` — imágenes limpias, sin contaminación (etiqueta 0)

El script `generar_dataset.py` convierte cada imagen al formato del dataset aplicando el siguiente pipeline:

1. **Redimensionado** a 128×128 píxeles (`cv2.INTER_AREA`).
2. **Conversión a escala de grises**.
3. **Normalización de iluminación** (opcional: CLAHE, corrección de gamma, ajuste de brillo/contraste).
4. **Suavizado** con filtro Gaussiano 5×5.
5. **Umbralización adaptativa Gaussiana** (por defecto) con bloque 31×31 y constante C=5.  
   Alternativa: umbral de Otsu global.
6. **Binarización** 0/255 → 0/1 (1 = fondo blanco, 0 = objeto presente).
7. **Aplanado** de la matriz 128×128 → vector de 16 384 valores.

La división train/test se hace con balance exacto de clases y mezcla aleatoria controlada por `SEMILLA = 42`.

---

## Variaciones registradas

- **Iluminación**: las condiciones de luz pueden variar entre sesiones de captura. El umbral adaptativo mitiga esto parcialmente; CLAHE puede activarse para mayor robustez.
- **Cámara/dispositivo**: no se documenta un único dispositivo de captura. Diferentes cámaras producen histogramas distintos que afectan el umbral.
- **Fondo**: se asume fondo predominantemente blanco. Fondos texturizados o de color pueden producir binarizaciones incorrectas.
- **Ángulo y distancia**: no se controlaron de forma estricta durante la captura.

---

## Limitaciones conocidas

- **Tamaño reducido**: 200 muestras de entrenamiento limitan la generalización. Los modelos presentan alta varianza entre ejecuciones.
- **Balance artificial**: el balance exacto 50/50 puede no reflejar la distribución real de producción.
- **Un solo tipo de contaminante**: el dataset solo contempla granos de arroz. Otros tipos de contaminantes no están representados.

---

## Reproducibilidad

Para regenerar el dataset exacto, debe primero tener en carpetas llamadas positive y negative las fotos correspondientes a cada uno. Después de esto, debe ajustar los siguientes parámetros y correr el programa.

```bash
# Asegurarse de que SEMILLA = 42, TRAIN_POR_CLASE = 100, TEST_POR_CLASE = 30
python generar_dataset.py
```
`TRAIN_POR_CLASE` y `TEST_POR_CLASE` permiten dividir los datos en dos datasets. El primero para entrenar el modelo y el segundo para evaluarlo con datos que nunca ha visto.

Las matrices de verificación de binarización se guardan automáticamente en `verificacion/` para inspección visual.

"""
Genera un dataset CSV a partir de imágenes en carpetas 'positive' y 'negative'.

Estructura esperada:
    positive/   <- imágenes con grano de arroz (contaminación)
    negative/   <- imágenes sin grano de arroz

Cada fila del CSV = 16 384 valores (128x128 píxeles aplanados) + etiqueta final.
Codificación de píxeles:
    1 = fondo completamente blanco
    0 = píxel con objeto presente
Etiqueta:
    1 = imagen positiva (contiene grano de arroz)
    0 = imagen negativa (no contiene grano de arroz)

Las matrices binarias de verificación se guardan en 'reports/figures/'.
"""

import os
import csv
import cv2
import numpy as np

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Configuración ─────────────────────────────────────────────────────────────
IMG_SIZE       = 128
CARPETA_POS    = os.path.join(_BASE_DIR, "data", "raw", "positive")
CARPETA_NEG    = os.path.join(_BASE_DIR, "data", "raw", "negative")
ARCHIVO_SALIDA = os.path.join(_BASE_DIR, "data", "processed", "dataset.csv")
CARPETA_VERIF  = os.path.join(_BASE_DIR, "reports", "figures")
EXTENSIONES    = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# ── División train / test ─────────────────────────────────────────────────────
# Ambos archivos tendrán exactamente la misma cantidad de positivos y negativos.
#   TRAIN_POR_CLASE : positivos en train  = negativos en train
#   TEST_POR_CLASE  : positivos en test   = negativos en test
TRAIN_POR_CLASE = 100   # → train.csv tendrá 200 filas  (100 pos + 100 neg)
TEST_POR_CLASE  = 30    # → test.csv  tendrá  60 filas  ( 30 pos +  30 neg)
SEMILLA         = 42    # semilla para reproducibilidad del shuffle
# ─────────────────────────────────────────────────────────────────────────────

# ── Ajuste de brillo / exposición ────────────────────────────────────────────
#   BRILLO    : desplazamiento de intensidad  [-255 … 255]
#               positivo = más claro, negativo = más oscuro
#   CONTRASTE : factor de escala de la imagen [0.0 … ∞]
#               1.0 = sin cambio, >1 aumenta contraste, <1 lo reduce
BRILLO    = 0
CONTRASTE = 1.0

# ── Normalización de iluminación variable ─────────────────────────────────────
# Problema: cada imagen puede tener condiciones de luz distintas, lo que hace
# que Otsu calcule umbrales inconsistentes entre imágenes.
#
# Soluciones disponibles (se pueden combinar):
#
#  1. CLAHE  — Contrast Limited Adaptive Histogram Equalization
#     Redistribuye el histograma de intensidades en bloques locales.
#     Normaliza el contraste de CADA imagen por separado antes de umbralizar.
#     → Recomendado para iluminación variable entre imágenes (tu caso).
#     CLAHE_CLIP  : límite de amplificación de contraste (2.0–4.0 típico)
#     CLAHE_GRID  : tamaño de la cuadrícula de tiles (8x8 típico)
#
#  2. GAMMA  — Corrección de gamma
#     Aclara (gamma<1) u oscurece (gamma>1) de forma no lineal.
#     Útil para imágenes muy oscuras o sobreexpuestas.
#     1.0 = sin corrección
#
#  3. UMBRAL ADAPTATIVO — en lugar de Otsu global
#     Calcula un umbral diferente para cada región de la imagen.
#     Útil cuando la iluminación varía DENTRO de una sola imagen.
#     Activar si las imágenes tienen sombras o gradientes de luz internos.

USAR_CLAHE          = False    # Recomendado: True
CLAHE_CLIP          = 4.0     # Límite de contraste CLAHE  [1.0–8.0]
CLAHE_GRID          = 8       # Tamaño de cuadrícula CLAHE [4, 8, 16]

USAR_GAMMA          = False   # Activar si las imágenes son muy oscuras/claras
GAMMA               = 1.0     # <1 aclara, >1 oscurece  (típico: 0.5–2.0)

USAR_UMBRAL_ADAPT   = True   # Activar si la luz varía dentro de cada imagen
ADAPT_BLOCK         = 31      # Tamaño del bloque local (impar, ≥11)
ADAPT_C             = 5       # Constante sustraída al umbral local
# ─────────────────────────────────────────────────────────────────────────────


def _tabla_gamma(gamma: float) -> np.ndarray:
    """Precalcula la tabla de lookup para corrección de gamma."""
    tabla = np.array([
        ((i / 255.0) ** (1.0 / gamma)) * 255
        for i in range(256)
    ], dtype=np.uint8)
    return tabla

# Tabla de gamma precalculada una sola vez
_GAMMA_TABLE = _tabla_gamma(GAMMA) if USAR_GAMMA and GAMMA != 1.0 else None
# Objeto CLAHE precreado una sola vez (costoso instanciar en cada imagen)
_CLAHE = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_GRID, CLAHE_GRID)) if USAR_CLAHE else None


def normalizar_iluminacion(gris: np.ndarray) -> np.ndarray:
    """
    Aplica la cadena de normalización de iluminación configurada:
      1. Brillo / contraste (ajuste lineal)
      2. Corrección de gamma  (si USAR_GAMMA=True)
      3. CLAHE               (si USAR_CLAHE=True)  ← recomendado para luz variable
    """
    # 1. Brillo y contraste/exposición
    if CONTRASTE != 1.0 or BRILLO != 0:
        gris = cv2.convertScaleAbs(gris, alpha=CONTRASTE, beta=BRILLO)

    # 2. Corrección de gamma
    if USAR_GAMMA and _GAMMA_TABLE is not None:
        gris = cv2.LUT(gris, _GAMMA_TABLE)

    # 3. CLAHE — normaliza histograma localmente, ecualizando la iluminación
    #    de cada imagen por separado sin importar sus condiciones de luz originales
    if USAR_CLAHE and _CLAHE is not None:
        gris = _CLAHE.apply(gris)

    return gris


def imagen_a_binario(ruta: str, ruta_verificacion: str) -> np.ndarray | None:
    """
    Lee una imagen, normaliza la iluminación, umbraliza y devuelve la
    matriz binaria 128×128 (valores 0/1).

    Guarda un PNG de verificación en 'ruta_verificacion'.

    Convención:
        1 → fondo blanco (sin objeto)
        0 → objeto presente (grano, mancha, etc.)
    """
    img = cv2.imread(ruta)
    if img is None:
        print(f"  [ADVERTENCIA] No se pudo leer: {ruta}")
        return None

    # Redimensionar a 128 × 128
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # Convertir a escala de grises
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Normalización de iluminación ───────────────────────────────────────
    gris = normalizar_iluminacion(gris)

    # ── Suavizado para reducir ruido antes de umbralizar ──────────────────
    blur = cv2.GaussianBlur(gris, (5, 5), 0)

    # ── Umbralización ─────────────────────────────────────────────────────
    if USAR_UMBRAL_ADAPT:
        # Umbral adaptativo: calcula un umbral diferente por región.
        # Útil cuando la iluminación varía dentro de una misma imagen.
        binaria = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            ADAPT_BLOCK, ADAPT_C
        )
    else:
        # Otsu: calcula un umbral global óptimo por imagen.
        # Con CLAHE activo funciona bien incluso con luces variables.
        _, binaria = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convertir 0/255 → 0/1
    binaria_01 = (binaria // 255).astype(np.uint8)

    # Convención: 1 = blanco (fondo), 0 = negro (objeto).
    # Otsu/adaptativo dejan el fondo claro en 255→1 y el objeto en 0.
    # Si tus imágenes tienen fondo oscuro y objetos claros, invierte:
    # binaria_01 = 1 - binaria_01

    # ── Guardar PNG de verificación (blanco=fondo, negro=objeto) ──────────
    img_verif = (binaria_01 * 255).astype(np.uint8)
    cv2.imwrite(ruta_verificacion, img_verif)

    return binaria_01          # matriz 128×128


def procesar_carpeta(carpeta: str, etiqueta: int, filas: list) -> int:
    """Procesa todas las imágenes de una carpeta y agrega filas al listado."""
    if not os.path.isdir(carpeta):
        print(f"[ERROR] Carpeta no encontrada: '{carpeta}'")
        return 0

    imagenes = [
        f for f in os.listdir(carpeta)
        if os.path.splitext(f)[1].lower() in EXTENSIONES
    ]

    if not imagenes:
        print(f"[AVISO] No se encontraron imágenes en '{carpeta}'")
        return 0

    # Subcarpeta de verificación separada por clase
    subverif = os.path.join(CARPETA_VERIF, os.path.basename(carpeta))
    os.makedirs(subverif, exist_ok=True)

    contador = 0
    for nombre in sorted(imagenes):
        ruta       = os.path.join(carpeta, nombre)
        nombre_png = os.path.splitext(nombre)[0] + "_bin.png"
        ruta_verif = os.path.join(subverif, nombre_png)

        matriz = imagen_a_binario(ruta, ruta_verif)
        if matriz is None:
            continue

        # Aplanar la matriz 128×128 → vector de 16 384 elementos + etiqueta
        vector = matriz.flatten()
        filas.append(np.append(vector, etiqueta))
        contador += 1

    print(f"  {carpeta}/: {contador} imágenes  (etiqueta={etiqueta})")
    return contador


def guardar_csv(ruta: str, filas: list[np.ndarray]) -> None:
    with open(ruta, "w", newline="") as f:
        writer = csv.writer(f)
        for fila in filas:
            writer.writerow(fila.astype(int).tolist())


def dividir_y_guardar(filas: list[np.ndarray]) -> None:
    """
    Divide las filas en train.csv y test.csv con balance exacto de clases.
    Cada archivo tiene TRAIN_POR_CLASE (o TEST_POR_CLASE) muestras por clase.
    """
    rng = np.random.default_rng(SEMILLA)

    # Separar por etiqueta (última columna)
    positivos = [f for f in filas if f[-1] == 1]
    negativos = [f for f in filas if f[-1] == 0]

    necesarios = TRAIN_POR_CLASE + TEST_POR_CLASE

    for clase, muestras in [("positivos", positivos), ("negativos", negativos)]:
        if len(muestras) < necesarios:
            print(f"  [ERROR] Se necesitan {necesarios} {clase} "
                  f"({TRAIN_POR_CLASE} train + {TEST_POR_CLASE} test) "
                  f"pero solo hay {len(muestras)}.")
            return

    # Mezclar cada clase de forma independiente
    idx_pos = rng.permutation(len(positivos))
    idx_neg = rng.permutation(len(negativos))

    pos_train = [positivos[i] for i in idx_pos[:TRAIN_POR_CLASE]]
    pos_test  = [positivos[i] for i in idx_pos[TRAIN_POR_CLASE:TRAIN_POR_CLASE + TEST_POR_CLASE]]
    neg_train = [negativos[i] for i in idx_neg[:TRAIN_POR_CLASE]]
    neg_test  = [negativos[i] for i in idx_neg[TRAIN_POR_CLASE:TRAIN_POR_CLASE + TEST_POR_CLASE]]

    # Mezclar positivos y negativos dentro de cada split
    train = pos_train + neg_train
    test  = pos_test  + neg_test
    rng.shuffle(train)
    rng.shuffle(test)

    _processed = os.path.join(_BASE_DIR, "data", "processed")
    guardar_csv(os.path.join(_processed, "train.csv"), train)
    guardar_csv(os.path.join(_processed, "test.csv"),  test)

    print(f"\n  train.csv → {len(train)} filas  "
          f"({TRAIN_POR_CLASE} pos + {TRAIN_POR_CLASE} neg)")
    print(f"  test.csv  → {len(test)} filas  "
          f"({TEST_POR_CLASE} pos + {TEST_POR_CLASE} neg)")


def main():
    os.makedirs(CARPETA_VERIF, exist_ok=True)

    filas = []

    print("=== Procesando imágenes ===")
    print(f"    Brillo={BRILLO:+d}  Contraste={CONTRASTE:.2f}  "
          f"CLAHE={'ON (clip={}, grid={}x{})'.format(CLAHE_CLIP, CLAHE_GRID, CLAHE_GRID) if USAR_CLAHE else 'OFF'}  "
          f"Gamma={'ON ({})'.format(GAMMA) if USAR_GAMMA else 'OFF'}  "
          f"UmbralAdapt={'ON' if USAR_UMBRAL_ADAPT else 'OFF'}\n")

    n_pos = procesar_carpeta(CARPETA_POS, etiqueta=1, filas=filas)
    n_neg = procesar_carpeta(CARPETA_NEG, etiqueta=0, filas=filas)

    total = n_pos + n_neg
    if total == 0:
        print("\nNo se procesaron imágenes. Verifica las carpetas 'positive' y 'negative'.")
        return

    print(f"\nTotal procesado: {total} imágenes  (positivas={n_pos}, negativas={n_neg})")
    print(f"Matrices de verificación guardadas en '{CARPETA_VERIF}/'")

    # Dataset completo
    print(f"\nGuardando dataset completo en '{ARCHIVO_SALIDA}' ...")
    guardar_csv(ARCHIVO_SALIDA, filas)

    # Splits balanceados
    print("Generando splits balanceados ...")
    dividir_y_guardar(filas)

    print(f"\nListo. Columnas por fila: {IMG_SIZE * IMG_SIZE + 1}  ({IMG_SIZE}x{IMG_SIZE} píxeles + etiqueta)")


if __name__ == "__main__":
    main()

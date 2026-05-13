"""
Procesa imágenes de una carpeta y genera predicciones con el modelo elegido.

Modos de operación (auto-detectados):

  Modo predicción  — la carpeta contiene imágenes directamente.
                     Guarda: imagen, prediccion, etiqueta, probabilidad.

  Modo evaluación  — la carpeta contiene subcarpetas 'positive/' y 'negative/'.
                     Usa el nombre de la subcarpeta como etiqueta real pero NO
                     se la pasa al modelo. Calcula métricas comparando predicción
                     vs etiqueta real y guarda el CSV completo.
"""

import os
import sys
import csv
import numpy as np
import joblib
from datetime import datetime

from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score
)

try:
    import cv2
except ImportError:
    print("[ERROR] OpenCV no instalado. Ejecuta: pip install opencv-python")
    sys.exit(1)

# Needed for joblib unpickling of saved model objects
import types as _types
import models as _models_module
from models import ThresholdedSVM, ThresholdedRF, RFSVMHybrid, SVMRFHybrid
_mw = _types.ModuleType("model_wrappers")
for _attr in ("ThresholdedSVM", "ThresholdedRF", "RFSVMHybrid", "SVMRFHybrid"):
    setattr(_mw, _attr, getattr(_models_module, _attr))
sys.modules.setdefault("model_wrappers", _mw)

# ── Configuración de imagen (debe coincidir con generate_dataset.py) ──────────
IMG_SIZE = 128

BRILLO    = 0
CONTRASTE = 1.0

USAR_CLAHE        = False
CLAHE_CLIP        = 4.0
CLAHE_GRID        = 8

USAR_GAMMA        = False
GAMMA             = 1.0

USAR_UMBRAL_ADAPT = True
ADAPT_BLOCK       = 31
ADAPT_C           = 5
# ─────────────────────────────────────────────────────────────────────────────

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

_BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_BASE_DIR, "models")

MODELS = {
    "1": ("Árbol de Decisión", os.path.join(_MODELS_DIR, "decision_tree_model.joblib"),         "sklearn"),
    "2": ("Naive Bayes",       os.path.join(_MODELS_DIR, "naive_bayes_model.joblib"),            "sklearn"),
    "3": ("KNN",               os.path.join(_MODELS_DIR, "knn_model.joblib"),                    "sklearn"),
    "4": ("SVM",               os.path.join(_MODELS_DIR, "c26797_sebastian_rojas.joblib"),       "sklearn"),
    "5": ("Random Forest",     os.path.join(_MODELS_DIR, "random_forest_model.joblib"),          "sklearn"),
    "6": ("Red Neuronal",      os.path.join(_MODELS_DIR, "nn_model_keras.keras"),                "keras"),
    "7": ("RF+SVM Híbrido",    os.path.join(_MODELS_DIR, "rf_svm_model.joblib"),                 "sklearn"),
    "8": ("SVM+RF Híbrido",    os.path.join(_MODELS_DIR, "svm_rf_model.joblib"),                 "sklearn"),
}

# Precalcular CLAHE y tabla gamma una sola vez
_CLAHE = (
    cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_GRID, CLAHE_GRID))
    if USAR_CLAHE else None
)
_GAMMA_TABLE = None
if USAR_GAMMA and GAMMA != 1.0:
    _GAMMA_TABLE = np.array(
        [((i / 255.0) ** (1.0 / GAMMA)) * 255 for i in range(256)], dtype=np.uint8
    )


def normalizar_iluminacion(gris: np.ndarray) -> np.ndarray:
    if CONTRASTE != 1.0 or BRILLO != 0:
        gris = cv2.convertScaleAbs(gris, alpha=CONTRASTE, beta=BRILLO)
    if USAR_GAMMA and _GAMMA_TABLE is not None:
        gris = cv2.LUT(gris, _GAMMA_TABLE)
    if USAR_CLAHE and _CLAHE is not None:
        gris = _CLAHE.apply(gris)
    return gris


def imagen_a_vector(ruta: str):
    """Lee una imagen, aplica binarización y devuelve vector de 16384 floats."""
    img = cv2.imread(ruta)
    if img is None:
        return None

    img  = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gris = normalizar_iluminacion(gris)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)

    if USAR_UMBRAL_ADAPT:
        binaria = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            ADAPT_BLOCK, ADAPT_C
        )
    else:
        _, binaria = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    binaria_01 = (binaria // 255).astype(np.float32)
    return binaria_01.flatten()   # shape (16384,)


def load_model(name, path, model_type):
    if not os.path.exists(path):
        print(f"  [ERROR] No se encontró: {path}")
        return None, None
    try:
        if model_type == "keras":
            try:
                import tensorflow as tf
            except ImportError:
                print("  [ERROR] TensorFlow no instalado.")
                return None, None
            model = tf.keras.models.load_model(path)
            import json as _json
            meta_path = path.replace(".keras", "_meta.json")
            model._threshold = 0.5
            if os.path.exists(meta_path):
                with open(meta_path) as _f:
                    model._threshold = float(_json.load(_f).get("threshold", 0.5))
                print(f"  [OK] {name} (Keras) cargado. umbral={model._threshold:.3f}")
            else:
                print(f"  [OK] {name} (Keras) cargado.")
            return model, "keras"
        else:
            model = joblib.load(path)
            print(f"  [OK] {name} (sklearn) cargado.")
            return model, "sklearn"
    except Exception as e:
        print(f"  [ERROR] Al cargar {path}: {e}")
        return None, None


def predict_single(model, model_type, x_vec: np.ndarray) -> tuple[int, float]:
    """Devuelve (prediccion_int, probabilidad_float)."""
    X = x_vec.reshape(1, -1)
    if model_type == "keras":
        proba     = float(model.predict(X, verbose=0).flatten()[0])
        threshold = getattr(model, "_threshold", 0.5)
        pred      = int(proba >= threshold)
        return pred, proba
    else:
        pred = int(model.predict(X)[0])
        proba = 0.5
        if hasattr(model, 'predict_proba'):
            try:
                proba = float(model.predict_proba(X)[0, 1])
            except Exception:
                pass
        return pred, proba


def select_model_menu() -> tuple[str, str, str]:
    print("\n" + "="*60)
    print("  SELECCIÓN DE MODELO")
    print("="*60)
    for key, (name, path, _) in MODELS.items():
        status = "existe" if os.path.exists(path) else "no encontrado"
        print(f"    [{key}] {name:<22} ({status})")
    print("="*60)
    choice = input("  Ingresa el número del modelo: ").strip()
    if choice not in MODELS:
        print("  Opción inválida. Saliendo.")
        sys.exit(1)
    name, path, mtype = MODELS[choice]
    return name, path, mtype


def recopilar_imagenes(carpeta: str) -> list[tuple[str, str, int | None]]:
    """
    Devuelve lista de (ruta_absoluta, nombre_archivo, etiqueta_real|None).

    Si la carpeta contiene 'positive/' y/o 'negative/', activa modo evaluación
    y asigna etiqueta 1 / 0 respectivamente.
    Si no, recoge imágenes directamente de la carpeta raíz sin etiqueta.
    """
    sub_pos = os.path.join(carpeta, "positive")
    sub_neg = os.path.join(carpeta, "negative")
    tiene_subs = os.path.isdir(sub_pos) or os.path.isdir(sub_neg)

    items = []
    if tiene_subs:
        for sub, etiqueta in [(sub_pos, 1), (sub_neg, 0)]:
            if not os.path.isdir(sub):
                continue
            for nombre in sorted(os.listdir(sub)):
                if os.path.splitext(nombre)[1].lower() in EXTENSIONES:
                    items.append((os.path.join(sub, nombre), nombre, etiqueta))
    else:
        for nombre in sorted(os.listdir(carpeta)):
            if os.path.splitext(nombre)[1].lower() in EXTENSIONES:
                items.append((os.path.join(carpeta, nombre), nombre, None))

    return items


def mostrar_metricas(y_true: list, y_pred: list, y_proba: list) -> None:
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    y_b = np.array(y_proba)

    recall_neg = recall_score(y_t, y_p, pos_label=0, zero_division=0)
    recall_pos = recall_score(y_t, y_p, pos_label=1, zero_division=0)

    print(f"\n{'='*60}")
    print("  MÉTRICAS DE EVALUACIÓN")
    print(f"{'='*60}")
    print(f"  Accuracy      : {accuracy_score(y_t, y_p):.4f}")
    print(f"  Precision     : {precision_score(y_t, y_p, zero_division=0):.4f}")
    print(f"  Recall Pos    : {recall_pos:.4f}")
    print(f"  Recall Neg    : {recall_neg:.4f}")
    print(f"  F1-Score      : {f1_score(y_t, y_p, zero_division=0):.4f}")
    try:
        print(f"  ROC-AUC       : {roc_auc_score(y_t, y_b):.4f}")
    except Exception:
        pass

    if recall_neg < 0.3 or recall_pos < 0.3:
        print("\n  SESGO DETECTADO:")
        if recall_neg < 0.3:
            print(f"      Recall Neg muy bajo ({recall_neg:.2f})")
        if recall_pos < 0.3:
            print(f"      Recall Pos muy bajo ({recall_pos:.2f})")

    print(f"\n{classification_report(y_t, y_p, target_names=['Negativo', 'Positivo'], zero_division=0)}")
    print("  Matriz de Confusión:")
    print(confusion_matrix(y_t, y_p))


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("   PREDICCIÓN POR CARPETA")
    print("="*60)

    # Carpeta de imágenes
    if len(sys.argv) > 1:
        carpeta = sys.argv[1]
    else:
        carpeta = input("\n  Ruta de la carpeta con imágenes: ").strip()

    if not os.path.isdir(carpeta):
        print(f"[ERROR] No se encontró la carpeta: {carpeta}")
        sys.exit(1)

    # Detectar modo
    items       = recopilar_imagenes(carpeta)
    modo_eval   = items and items[0][2] is not None
    modo_nombre = "EVALUACIÓN (con etiquetas)" if modo_eval else "PREDICCIÓN (sin etiquetas)"
    print(f"\n  Modo detectado: {modo_nombre}")
    if modo_eval:
        n_pos = sum(1 for _, _, e in items if e == 1)
        n_neg = sum(1 for _, _, e in items if e == 0)
        print(f"  Imágenes: {n_pos} positivas | {n_neg} negativas")
    else:
        print(f"  Imágenes encontradas: {len(items)}")

    if not items:
        print("\n[AVISO] No se encontraron imágenes.")
        sys.exit(0)

    # Modelo
    model_name, model_path, model_type = select_model_menu()
    model, loaded_type = load_model(model_name, model_path, model_type)
    if model is None:
        sys.exit(1)

    print(f"\n  Procesando {len(items)} imágenes...\n")

    resultados = []   # filas para el CSV
    errores    = 0
    y_true_all = []
    y_pred_all = []
    y_proba_all= []

    for ruta, nombre, etiqueta_real in items:
        vec = imagen_a_vector(ruta)
        if vec is None:
            print(f"  [SKIP] No se pudo leer: {nombre}")
            errores += 1
            continue

        # Solo el vector de píxeles va al modelo — la etiqueta nunca se pasa
        pred, proba = predict_single(model, loaded_type, vec)
        pred_label  = "Positivo" if pred == 1 else "Negativo"

        if modo_eval:
            real_label = "Positivo" if etiqueta_real == 1 else "Negativo"
            correcto   = "OK" if pred == etiqueta_real else "FAIL"
            print(f"  {nombre:<38} real={real_label:<9} pred={pred_label:<9} {correcto}")
            resultados.append((nombre, etiqueta_real, real_label, pred, pred_label, round(proba, 4)))
            y_true_all.append(etiqueta_real)
            y_pred_all.append(pred)
            y_proba_all.append(proba)
        else:
            print(f"  {nombre:<40} → {pred_label}  (prob={proba:.4f})")
            resultados.append((nombre, pred, pred_label, round(proba, 4)))

    # Métricas (solo modo evaluación)
    if modo_eval and y_true_all:
        mostrar_metricas(y_true_all, y_pred_all, y_proba_all)

    # Guardar CSV
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_nombre = f"predicciones_{model_name.replace(' ', '_')}_{timestamp}.csv"
    csv_path   = os.path.join(_BASE_DIR, csv_nombre)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if modo_eval:
            writer.writerow(["imagen", "etiqueta_real", "real_label",
                             "prediccion", "pred_label", "probabilidad"])
        else:
            writer.writerow(["imagen", "prediccion", "etiqueta", "probabilidad"])
        writer.writerows(resultados)

    print(f"\n{'='*60}")
    print(f"  Total procesadas : {len(resultados)}")
    print(f"  Positivos pred   : {sum(1 for r in resultados if r[3 if modo_eval else 1] == 1)}")
    print(f"  Negativos pred   : {sum(1 for r in resultados if r[3 if modo_eval else 1] == 0)}")
    if errores:
        print(f"  Imágenes con error: {errores}")
    print(f"\n  Resultados guardados: {csv_path}")
    print("="*60)

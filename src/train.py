import os

import numpy as np
import pandas as pd
import joblib

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, balanced_accuracy_score,
    recall_score, roc_auc_score
)

from models import ThresholdedRF

_BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(_BASE_DIR, "data", "processed", "train.csv")
MODEL_PATH = os.path.join(_BASE_DIR, "models", "svm_model.joblib")

# Parámetros extraídos de c26797_sebastian_rojas.joblib
N_COMPONENTS  = 15
C             = 36.36
GAMMA         = 0.002
TARGET_RECALL = 0.90


def geometric_recall(y_true, y_pred):
    r0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    r1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    return (r0 * r1) ** 0.5


# ============================================================
# Cargar datos
# ============================================================
print("Cargando dataset...")
df = pd.read_csv(DATA_PATH, header=None)
df = df.apply(pd.to_numeric, errors="coerce").dropna()

X = df.iloc[:, :16384].values.astype(np.float32)
y = df.iloc[:, 16384].values.astype(int)

print(f"Shape X: {X.shape} | Positivos: {y.sum()} | Negativos: {(1-y).sum()}")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Val: {len(X_val)}\n")


# ============================================================
# Entrenar
# ============================================================
print(f"Entrenando SVM (C={C}, gamma={GAMMA}, n_components={N_COMPONENTS})...")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca",    PCA(n_components=N_COMPONENTS, random_state=42)),
    ("svm",    SVC(kernel="rbf", C=C, gamma=GAMMA,
                   probability=True, class_weight="balanced",
                   random_state=42)),
])
pipeline.fit(X_train, y_train)
print("Entrenamiento completado.")


# ============================================================
# Búsqueda de umbral óptimo en validación
# ============================================================
y_proba = pipeline.predict_proba(X_val)[:, 1]

best_thresh = 0.5
best_geo    = 0.0
for t in np.linspace(0.05, 0.95, 181):
    pred_t = (y_proba >= t).astype(int)
    geo    = geometric_recall(y_val, pred_t)
    if geo > best_geo:
        best_geo    = geo
        best_thresh = t

print(f"\nUmbral óptimo (val): {best_thresh:.3f}  |  geometric_recall: {best_geo:.4f}")


# ============================================================
# Evaluación final
# ============================================================
y_pred     = (y_proba >= best_thresh).astype(int)
recall_neg = recall_score(y_val, y_pred, pos_label=0, zero_division=0)
recall_pos = recall_score(y_val, y_pred, pos_label=1, zero_division=0)

unique, counts = np.unique(y_pred, return_counts=True)
dist  = dict(zip(unique, counts))
total = len(y_pred)

print(f"\nDistribución de predicciones:")
print(f"  Negativos: {dist.get(0, 0)} ({dist.get(0, 0)/total*100:.1f}%)")
print(f"  Positivos: {dist.get(1, 0)} ({dist.get(1, 0)/total*100:.1f}%)")

if recall_neg >= TARGET_RECALL and recall_pos >= TARGET_RECALL:
    print(f"  META ALCANZADA: ambos recalls >= {TARGET_RECALL:.0%}")
elif min(recall_neg, recall_pos) >= 0.80:
    print("  ACEPTABLE: ambos recalls >= 80%, pero por debajo del 90%.")
else:
    print("  ADVERTENCIA: uno o ambos recalls por debajo del 80%.")

print(f"\nRecall Negativos  : {recall_neg:.4f}")
print(f"Recall Positivos  : {recall_pos:.4f}")
print(f"Accuracy          : {accuracy_score(y_val, y_pred):.4f}")
print(f"Balanced Accuracy : {balanced_accuracy_score(y_val, y_pred):.4f}")
try:
    print(f"ROC-AUC           : {roc_auc_score(y_val, y_proba):.4f}")
except Exception:
    pass
print(f"\n{classification_report(y_val, y_pred, target_names=['Negativo', 'Positivo'], zero_division=0)}")
print("Matriz de Confusión:")
print(confusion_matrix(y_val, y_pred))


# ============================================================
# Guardar
# ============================================================
model = ThresholdedRF(pipeline=pipeline, threshold=best_thresh)
joblib.dump(model, MODEL_PATH)
print(f"\nModelo guardado: {MODEL_PATH}  (umbral={best_thresh:.3f})")

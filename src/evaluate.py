import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# Needed for joblib unpickling of saved model objects
from models import ThresholdedSVM, ThresholdedRF, RFSVMHybrid, SVMRFHybrid  # noqa: F401

N_FEATURES = 16384

_BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_BASE_DIR, "models")

MODELS = {
    "1": ("SVM", os.path.join(_MODELS_DIR, "c26797_sebastian_rojas.joblib"), "sklearn"),
}


def load_dataset(path, has_labels):
    print(f"\nCargando dataset: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    df = pd.read_csv(path, header=None)
    df = df.apply(pd.to_numeric, errors='coerce')
    filas_antes = len(df)
    df = df.dropna()
    eliminadas = filas_antes - len(df)
    if eliminadas > 0:
        print(f"  Filas eliminadas (NaN o headers de texto): {eliminadas}")

    X = df.iloc[:, :N_FEATURES].values.astype(np.float32)

    if has_labels:
        y = df.iloc[:, N_FEATURES].values.astype(int)
        print(f"  {len(X)} muestras | Positivos: {y.sum()} | Negativos: {(1-y).sum()}")
        return X, y
    else:
        print(f"  {len(X)} muestras (sin etiquetas)")
        return X, None


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
                print(f"  [OK] {name} (Keras) cargado.  umbral={model._threshold:.3f}")
            else:
                print(f"  [OK] {name} (Keras) cargado.")
            return model, "keras"
        else:
            model = joblib.load(path)
            print(f"  [OK] {name} (sklearn) cargado.")
            return model, "sklearn"
    except Exception as e:
        print(f"  [ERROR] No se pudo cargar {path}: {e}")
        return None, None


def predict(model, model_type, X):
    if model_type == "keras":
        proba     = model.predict(X, verbose=0).flatten()
        threshold = getattr(model, "_threshold", 0.5)
        y_pred    = (proba >= threshold).astype(int)
        return y_pred, proba
    else:
        y_pred = model.predict(X)
        proba  = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(X)[:, 1]
            except Exception:
                pass
        return y_pred, proba


def evaluate_model(name, model, model_type, X, y):
    print(f"\n{'='*60}")
    print(f"  Modelo: {name}")
    print(f"{'='*60}")

    y_pred, y_proba = predict(model, model_type, X)

    auc_score = None
    if y is not None and y_proba is not None:
        try:
            auc_score = roc_auc_score(y, y_proba)
        except Exception:
            pass

    metrics = {}
    if y is not None:
        recall_neg = recall_score(y, y_pred, pos_label=0, zero_division=0)
        recall_pos = recall_score(y, y_pred, pos_label=1, zero_division=0)

        metrics = {
            "accuracy"   : accuracy_score(y, y_pred),
            "precision"  : precision_score(y, y_pred, zero_division=0),
            "recall_pos" : recall_pos,
            "recall_neg" : recall_neg,
            "f1"         : f1_score(y, y_pred, zero_division=0),
            "roc_auc"    : auc_score if auc_score else "N/A",
        }

        print(f"  Accuracy      : {metrics['accuracy']:.4f}")
        print(f"  Precision     : {metrics['precision']:.4f}")
        print(f"  Recall Pos    : {recall_pos:.4f}")
        print(f"  Recall Neg    : {recall_neg:.4f}")
        print(f"  F1-Score      : {metrics['f1']:.4f}")
        if auc_score:
            print(f"  ROC-AUC       : {auc_score:.4f}")

        if recall_neg < 0.3 or recall_pos < 0.3:
            print(f"\n  SESGO DETECTADO:")
            if recall_neg < 0.3:
                print(f"      Recall Negativos muy bajo ({recall_neg:.2f})")
            if recall_pos < 0.3:
                print(f"      Recall Positivos muy bajo ({recall_pos:.2f})")

        print(f"\n{classification_report(y, y_pred, target_names=['Negativo', 'Positivo'])}")
        print("  Matriz de Confusión:")
        print(confusion_matrix(y, y_pred))
    else:
        unique, counts = np.unique(y_pred, return_counts=True)
        dist = dict(zip(unique, counts))
        print(f"  Negativos predichos: {dist.get(0, 0)}")
        print(f"  Positivos predichos: {dist.get(1, 0)}")

    return y_pred, metrics


def save_results(results_list, y_true, output_prefix="resultados"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Predicciones
    df = pd.DataFrame()
    if y_true is not None:
        df["esperado"]       = y_true
        df["esperado_label"] = np.where(y_true == 1, "Positivo", "Negativo")
    for name, y_pred, _ in results_list:
        col = name.replace(" ", "_")
        df[f"predicho_{col}"]       = y_pred
        df[f"predicho_label_{col}"] = np.where(y_pred == 1, "Positivo", "Negativo")
    pred_path = f"{output_prefix}_predicciones_{timestamp}.csv"
    df.to_csv(pred_path, index=False)
    print(f"\n  Predicciones guardadas : {pred_path}")

    # Métricas
    rows = []
    for name, _, metrics in results_list:
        if metrics:
            row = {"modelo": name}
            row.update(metrics)
            rows.append(row)
    df_m = pd.DataFrame(rows)
    if not df_m.empty:
        metrics_path = f"{output_prefix}_metricas_{timestamp}.csv"
        df_m.to_csv(metrics_path, index=False)
        print(f"  Métricas guardadas     : {metrics_path}")


def plot_confusion_matrices(results_list, y_true):
    valid = [(n, p) for n, p, _ in results_list if p is not None]
    if not valid or y_true is None:
        return

    cols = min(2, len(valid))
    rows = (len(valid) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).flatten()

    for ax, (name, y_pred) in zip(axes, valid):
        cm   = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Negativo", "Positivo"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=12, fontweight='bold')

    for ax in axes[len(valid):]:
        ax.set_visible(False)

    plt.suptitle("Matrices de Confusión — SVM vs RF+SVM", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("confusion_matrices_test3.png", dpi=150, bbox_inches='tight')
    print(f"  Gráfica guardada       : confusion_matrices_test3.png")
    plt.show()


def plot_comparison(results_list):
    rows = [(n, m) for n, _, m in results_list if m]
    if len(rows) < 2:
        return

    names   = [n for n, _ in rows]
    metrics = [m for _, m in rows]
    keys    = ["accuracy", "precision", "recall_pos", "recall_neg", "f1"]
    labels  = ["Accuracy", "Precision", "Recall Pos", "Recall Neg", "F1"]
    colors  = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']

    x     = np.arange(len(names))
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (key, label, color) in enumerate(zip(keys, labels, colors)):
        vals = [m.get(key, 0) if isinstance(m.get(key), float) else 0 for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}", ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.8, color='green', linestyle='--', linewidth=1, label='Objetivo (0.80)')
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("SVM vs RF+SVM — Comparación de Métricas", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("metrics_test3.png", dpi=150)
    print(f"  Gráfica guardada       : metrics_test3.png")
    plt.show()


# ─── Main ────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("   TEST3 — SVM  vs  RF+SVM HÍBRIDO")
    print("="*60)

    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        has_labels   = True
    else:
        path = input("\n  Ruta del dataset CSV (Enter para 'dataset_test.csv'): ").strip()
        dataset_path = path if path else "dataset_test.csv"
        print("\n  ¿El dataset incluye etiqueta real?")
        print("    [1] Sí   [2] No")
        has_labels = input("  Opción: ").strip() != "2"

    X, y = load_dataset(dataset_path, has_labels)

    print("\n" + "="*60)
    print("  Modelos disponibles:")
    for key, (name, path, _) in MODELS.items():
        status = "existe" if os.path.exists(path) else "no encontrado"
        print(f"    [{key}] {name:<22} ({status})")
    print("    [A] Probar AMBOS")
    print("="*60)
    choice = input("  Ingresa opción: ").strip().upper()

    if choice == "A":
        selected = [(name, path, mtype) for _, (name, path, mtype) in MODELS.items()]
    elif choice in MODELS:
        name, path, mtype = MODELS[choice]
        selected = [(name, path, mtype)]
    else:
        print("  Opción inválida — probando ambos.")
        selected = [(name, path, mtype) for _, (name, path, mtype) in MODELS.items()]

    results_list = []
    for name, path, mtype in selected:
        model, loaded_type = load_model(name, path, mtype)
        if model is None:
            continue
        y_pred, metrics = evaluate_model(name, model, loaded_type, X, y)
        results_list.append((name, y_pred, metrics))

    if not results_list:
        print("\n  No se pudo cargar ningún modelo.")
        sys.exit(1)

    if y is not None:
        print(f"\n{'='*60}")
        print("  RESUMEN FINAL")
        print(f"{'='*60}")
        print(f"  {'Modelo':<22} {'Accuracy':>9} {'Prec':>6} {'Rec+':>6} {'Rec-':>6} {'F1':>6}")
        print(f"  {'-'*58}")
        for name, _, metrics in results_list:
            if metrics:
                rec_pos = metrics.get('recall_pos', 0)
                rec_neg = metrics.get('recall_neg', 0)
                flag    = " ⚠️" if rec_pos < 0.3 or rec_neg < 0.3 else " ✅"
                print(f"  {name:<22}"
                      f"{metrics['accuracy']:>9.4f}"
                      f"{metrics['precision']:>6.2f}"
                      f"{rec_pos:>6.2f}"
                      f"{rec_neg:>6.2f}"
                      f"{metrics['f1']:>6.2f}"
                      f"{flag}")

    save_results(results_list, y, output_prefix="test3_resultados")

    if y is not None:
        plot_confusion_matrices(results_list, y)
        if len(results_list) > 1:
            plot_comparison(results_list)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from m5b_pideeponet_model import MemmapCSIDataset, PIDeepONet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def map_people_to_band(y):
    y = np.asarray(y, dtype=np.int64)
    out = np.empty_like(y)
    out[(y >= 0) & (y <= 2)] = 0
    out[(y >= 3) & (y <= 5)] = 1
    out[(y >= 6) & (y <= 8)] = 2
    return out

def _evaluate_split(model, loader, model_type):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            x_b = batch[0].to(DEVICE)
            y_people = batch[1].to(DEVICE)

            if model_type.lower() == "pideeponet":
                logits, _ = model(x_b)
            else:
                logits = model(x_b)

            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_people.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_targets), np.array(all_preds), np.array(all_probs)

def run_post_hoc_diagnostics(model_path, model_type="pideeponet", output_dir=None, run_dir=None, num_classes=3):
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"⚠️ Aviso: No se encontró el modelo en {model_path} para el diagnóstico.")
        return

    # Resolución del directorio de salida en outputs/
    if output_dir is not None:
        diagnostics_dir = Path(output_dir) / "diagnostics"
    elif run_dir is not None:
        diagnostics_dir = Path(run_dir) / "diagnostics"
    else:
        project_dir = model_path.parent.parent
        diagnostics_dir = project_dir / "output" / "diagnostics"

    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📊 Guardando diagnósticos completos en: {diagnostics_dir}")

    project_dir = model_path.parent.parent
    tensors_dir = project_dir / "data" / "processed_tensors"
    meta_path = tensors_dir / "dataset_mc1_rx1_80mhz_meta.npz"

    meta = np.load(meta_path, allow_pickle=True)

    if num_classes == 3:
        tr_y = map_people_to_band(meta['train_y_people'])
        va_y = map_people_to_band(meta['val_y_people'])
        te_y = map_people_to_band(meta['test_y_people'])
        class_names = ["Baja (0-2)", "Media (3-5)", "Alta (6-8)"]
    else:
        tr_y = meta['train_y_people']
        va_y = meta['val_y_people']
        te_y = meta['test_y_people']
        num_classes = len(np.unique(meta['train_y_people']))
        class_names = [str(i) for i in range(num_classes)]

    # Cargar Datasets de los 3 splits
    datasets = {
        "Train": MemmapCSIDataset(tensors_dir / "X_train_frames.dat", meta['tr_shape'], meta['train_starts'], tr_y, meta['train_y_empty']),
        "Val":   MemmapCSIDataset(tensors_dir / "X_val_frames.dat", meta['va_shape'], meta['val_starts'], va_y, meta['val_y_empty']),
        "Test":  MemmapCSIDataset(tensors_dir / "X_test_frames.dat", meta['te_shape'], meta['test_starts'], te_y, meta['test_y_empty'])
    }

    # Cargar Arquitectura
    if model_type.lower() == "pideeponet":
        model = PIDeepONet(num_classes=num_classes, latent_dim=128).to(DEVICE)
    else:
        from m6_cnn_baseline_model import PureCNN2DBaseline
        model = PureCNN2DBaseline(num_classes=num_classes).to(DEVICE)

    model.load_state_dict(torch.load(model_path))

    # --- 1. GENERACIÓN DE LAS 3 MATRICES DE CONFUSIÓN Y REPORTES ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    summary_report_str = "=== REPORTE DETALLADO DE DIAGNÓSTICO (TRAIN / VAL / TEST) ===\n\n"

    test_y_true, test_y_probs = None, None

    for idx, (split_name, dataset) in enumerate(datasets.items()):
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
        y_true, y_pred, y_probs = _evaluate_split(model, loader, model_type)

        if split_name == "Test":
            test_y_true = y_true
            test_y_probs = y_probs

        # Generar Matriz
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', cbar=False,
                    xticklabels=class_names, yticklabels=class_names, ax=axes[idx])
        axes[idx].set_title(f'Matriz de Confusión - {split_name}')
        axes[idx].set_xlabel('Predicción del Modelo')
        axes[idx].set_ylabel('Real')

        # Acumular Métricas
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        pred_counts = np.bincount(y_pred, minlength=num_classes)

        summary_report_str += f"=========================================\n"
        summary_report_str += f" SPLIT: {split_name.upper()}\n"
        summary_report_str += f"=========================================\n"
        summary_report_str += classification_report(y_true, y_pred, target_names=class_names, digits=4)
        summary_report_str += f"--- MÉTRICAS CONSENSUS ({split_name}) ---\n"
        summary_report_str += f"macro_f1 = {macro_f1:.4f}\n"
        summary_report_str += f"balanced_acc = {balanced_acc:.4f}\n"
        summary_report_str += f"pred_counts = {pred_counts}\n\n"

    plt.tight_layout()
    plt.savefig(diagnostics_dir / "confusion_matrices_all_splits.png", dpi=300)
    plt.close()

    # --- 2. GENERACIÓN DE CURVA ROC MULTI-CLASE (TEST TARGET) ---
    y_true_bin = label_binarize(test_y_true, classes=list(range(num_classes)))
    plt.figure(figsize=(8, 6))

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], test_y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Clase {class_names[i]} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title(f'Curva ROC Multi-Clase (Test Target - {model_path.stem})')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(diagnostics_dir / "roc_curve_test.png", dpi=300)
    plt.close()

    # Guardar reporte escrito acumulado
    with open(diagnostics_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(summary_report_str)

    print("✅ Reporte multi-split y métricas Consensus generadas.")
    print(f"✅ Matrices de confusión (3 splits) guardadas en: {diagnostics_dir / 'confusion_matrices_all_splits.png'}")
    print(f"✅ Curva ROC guardada en: {diagnostics_dir / 'roc_curve_test.png'}")

import os
import time
import random
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

# Módulos del proyecto
from m5b_pideeponet_model import MemmapCSIDataset, PIDeepONet, compute_pi_loss
from experiment_logger import ExperimentLogger
from evaluate_diagnostics import run_post_hoc_diagnostics

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
TENSORS_DIR = DATA_DIR / "processed_tensors"
META_PATH = TENSORS_DIR / "dataset_mc1_rx1_80mhz_meta.npz"
CHECKPOINT_DIR = PROJECT_DIR / "models"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 256
NUM_WORKERS = 4
EPOCHS = 15
LEARNING_RATE = 1e-3
LAMBDA_PHYS = 0.01
LAMBDA_TEMP = 0.001  # 👈 Coeficiente de regularización temporal
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epoch(model, dataloader, optimizer, device, lambda_phys, lambda_temp, class_weights_tensor):
    model.train()
    running_loss = 0.0
    running_loss_temp = 0.0
    correct = 0
    total = 0

    for x_b, y_people, y_empty in dataloader:
        x_b = x_b.to(device)
        y_people = y_people.to(device)
        y_empty = y_empty.to(device)

        optimizer.zero_grad()
        logits, latent_field = model(x_b)

        # 🚀 Cómputo de pérdida con regularización temporal
        loss, _, _, loss_temp = compute_pi_loss(
            logits, latent_field, y_people, y_empty, 
            lambda_phys=lambda_phys, class_weights=class_weights_tensor, lambda_temp=lambda_temp
        )
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x_b.size(0)
        running_loss_temp += loss_temp.item() * x_b.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_people).sum().item()
        total += y_people.size(0)

    return running_loss / total, (correct / total) * 100.0, running_loss_temp / total

def evaluate(model, dataloader, device, lambda_phys, lambda_temp, class_weights_tensor):
    model.eval()
    running_loss = 0.0
    running_loss_temp = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x_b, y_people, y_empty in dataloader:
            x_b = x_b.to(device)
            y_people = y_people.to(device)
            y_empty = y_empty.to(device)

            logits, latent_field = model(x_b)
            loss, _, _, loss_temp = compute_pi_loss(
                logits, latent_field, y_people, y_empty, 
                lambda_phys=lambda_phys, class_weights=class_weights_tensor, lambda_temp=lambda_temp
            )

            running_loss += loss.item() * x_b.size(0)
            running_loss_temp += loss_temp.item() * x_b.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_people).sum().item()
            total += y_people.size(0)

    return running_loss / total, (correct / total) * 100.0, running_loss_temp / total

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento PI-DeepONet (9 Clases + Regularización Temporal)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    args = parser.parse_args()

    set_seed(args.seed)

    print("=== MÓDULO 6: PI-DEEPONET (9 CLASES CON PESOS BALANCEADOS Y REGULARIZACIÓN TEMPORAL) ===")
    print(f"Dispositivo activo: {DEVICE} ({torch.cuda.get_device_name(0)}) | Semilla: {args.seed}")

    logger = ExperimentLogger(experiment_name=f"PI_DeepONet_9Class_TempLoss_Seed_{args.seed}", base_dir=PROJECT_DIR / "runs")

    data_files = {
        "metadata": META_PATH,
        "X_train": TENSORS_DIR / "X_train_frames.dat",
        "X_val": TENSORS_DIR / "X_val_frames.dat",
        "X_test": TENSORS_DIR / "X_test_frames.dat"
    }

    if not META_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo de metadata en {META_PATH}")

    meta = np.load(META_PATH, allow_pickle=True)

    # 🚀 Cargar etiquetas crudas originales (0 a 8 personas)
    train_y = meta['train_y_people']
    val_y   = meta['val_y_people']
    test_y  = meta['test_y_people']
    num_classes = len(np.unique(train_y))

    hyperparameters = {
        "SEED": args.seed,
        "NUM_CLASSES": num_classes,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_WORKERS": NUM_WORKERS,
        "EPOCHS": EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "LAMBDA_PHYS": LAMBDA_PHYS,
        "LAMBDA_TEMP": LAMBDA_TEMP,
        "MODEL": "PI-DeepONet_9Class_TempLoss",
        "DEVICE": str(DEVICE)
    }

    logger.start_experiment(hyperparameters=hyperparameters, data_files=data_files)

    # 🚀 Calcular pesos para las 9 clases
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_y),
        y=train_y
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    print(f"⚖️ Pesos de 9 clases calculados: {class_weights}")

    test_sets = np.unique(meta['test_set_ids'])

    train_dataset = MemmapCSIDataset(TENSORS_DIR / "X_train_frames.dat", meta['tr_shape'], meta['train_starts'], train_y, meta['train_y_empty'])
    val_dataset   = MemmapCSIDataset(TENSORS_DIR / "X_val_frames.dat", meta['va_shape'], meta['val_starts'], val_y, meta['val_y_empty'])
    test_dataset  = MemmapCSIDataset(TENSORS_DIR / "X_test_frames.dat", meta['te_shape'], meta['test_starts'], test_y, meta['test_y_empty'])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = PIDeepONet(num_classes=num_classes, latent_dim=128).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    best_model_path = CHECKPOINT_DIR / f"pideeponet_9class_temploss_seed_{args.seed}_best.pth"

    print("Iniciando entrenamiento con pérdida física + temporal...")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_loss_temp = train_epoch(model, train_loader, optimizer, DEVICE, LAMBDA_PHYS, LAMBDA_TEMP, class_weights_tensor)
        val_loss, val_acc, val_loss_temp = evaluate(model, val_loader, DEVICE, LAMBDA_PHYS, LAMBDA_TEMP, class_weights_tensor)

        scheduler.step(val_loss)

        # 🚀 Registrar las pérdidas temporales independientemente en el logger
        extra_metrics = {
            "tr_loss_temp": round(tr_loss_temp, 6),
            "val_loss_temp": round(val_loss_temp, 6)
        }
        logger.log_epoch(epoch, tr_loss, tr_acc, val_loss, val_acc, extra_metrics=extra_metrics)

        print(f"Época [{epoch:02d}/{EPOCHS:02d}] | Train Loss: {tr_loss:.4f} (L_temp: {tr_loss_temp:.6f}) - Acc: {tr_acc:.2f}% | Val Loss: {val_loss:.4f} (L_temp: {val_loss_temp:.6f}) - Acc: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    print("\nCargando mejor modelo para evaluación Cross-Domain...")
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, _ = evaluate(model, test_loader, DEVICE, LAMBDA_PHYS, LAMBDA_TEMP, class_weights_tensor)

    test_metrics = {
        "Target_Domain": str(test_sets[0]),
        "Loss": float(test_loss),
        "Accuracy_Percent": float(test_acc)
    }

    logger.end_experiment(test_metrics=test_metrics)

    # 🚀 Generar diagnósticos para las 9 clases
    run_post_hoc_diagnostics(best_model_path, model_type="pideeponet", output_dir=logger.output_dir, num_classes=num_classes)

if __name__ == "__main__":
    main()

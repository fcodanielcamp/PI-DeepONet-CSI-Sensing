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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epoch(model, dataloader, optimizer, device, lambda_phys, lambda_temp, lambda_energy, gamma, class_weights_tensor):
    model.train()
    running_loss = 0.0
    running_loss_temp = 0.0
    running_loss_energy = 0.0
    correct = 0
    total = 0

    for x_b, y_people, y_empty in dataloader:
        x_b = x_b.to(device)
        y_people = y_people.to(device)
        y_empty = y_empty.to(device)

        optimizer.zero_grad()
        logits, latent_field = model(x_b)

        # Cómputo de pérdida con CORAL ordinal, regularización temporal y energía RCS
        loss, _, _, loss_temp, loss_energy = compute_pi_loss(
            logits, latent_field, y_people, y_empty,
            lambda_phys=lambda_phys, class_weights=class_weights_tensor, 
            lambda_temp=lambda_temp, lambda_energy=lambda_energy, gamma=gamma
        )
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x_b.size(0)
        running_loss_temp += loss_temp.item() * x_b.size(0)
        running_loss_energy += loss_energy.item() * x_b.size(0)

        # Predicción Ordinal CORAL: contar umbrales binarios superados (sigmoid > 0.5)[cite: 2]
        probs = torch.sigmoid(logits)
        preds = torch.sum(probs > 0.5, dim=1)

        correct += (preds == y_people).sum().item()
        total += y_people.size(0)

    return running_loss / total, (correct / total) * 100.0, running_loss_temp / total, running_loss_energy / total

def evaluate(model, dataloader, device, lambda_phys, lambda_temp, lambda_energy, gamma, class_weights_tensor):
    model.eval()
    running_loss = 0.0
    running_loss_temp = 0.0
    running_loss_energy = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x_b, y_people, y_empty in dataloader:
            x_b = x_b.to(device)
            y_people = y_people.to(device)
            y_empty = y_empty.to(device)

            logits, latent_field = model(x_b)
            loss, _, _, loss_temp, loss_energy = compute_pi_loss(
                logits, latent_field, y_people, y_empty,
                lambda_phys=lambda_phys, class_weights=class_weights_tensor, 
                lambda_temp=lambda_temp, lambda_energy=lambda_energy, gamma=gamma
            )

            running_loss += loss.item() * x_b.size(0)
            running_loss_temp += loss_temp.item() * x_b.size(0)
            running_loss_energy += loss_energy.item() * x_b.size(0)

            # Predicción Ordinal CORAL[cite: 2]
            probs = torch.sigmoid(logits)
            preds = torch.sum(probs > 0.5, dim=1)

            correct += (preds == y_people).sum().item()
            total += y_people.size(0)

    return running_loss / total, (correct / total) * 100.0, running_loss_temp / total, running_loss_energy / total

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento PI-DeepONet CORAL + Física (9 Clases Ordinales)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--lambda_phys", type=float, default=0.01, help="Coeficiente para pérdida de estado vacío")
    parser.add_argument("--lambda_temp", type=float, default=0.001, help="Coeficiente para regularización temporal")
    parser.add_argument("--lambda_energy", type=float, default=0.001, help="Coeficiente para regularización de energía RCS")
    parser.add_argument("--gamma", type=float, default=1.0, help="Factor de escala para la energía dispersada")
    args = parser.parse_args()

    set_seed(args.seed)

    print("=== MÓDULO 6: PI-DEEPONET CORAL + FÍSICA (9 CLASES + TEMP + ENERGÍA) ===")
    print(f"Dispositivo activo: {DEVICE} ({torch.cuda.get_device_name(0)}) | Semilla: {args.seed}")
    print(f"Parámetros de Pérdida -> Lambda Phys: {args.lambda_phys} | Lambda Temp: {args.lambda_temp} | Lambda Energy: {args.lambda_energy} | Gamma: {args.gamma}")

    # Nombre único del experimento para mantener la trazabilidad
    exp_name = f"PI_DeepONet_CORAL_Lphys_{args.lambda_phys}_Ltemp_{args.lambda_temp}_Lenergy_{args.lambda_energy}_Seed_{args.seed}"
    logger = ExperimentLogger(experiment_name=exp_name, base_dir=PROJECT_DIR / "runs")

    data_files = {
        "metadata": META_PATH,
        "X_train": TENSORS_DIR / "X_train_frames.dat",
        "X_val": TENSORS_DIR / "X_val_frames.dat",
        "X_test": TENSORS_DIR / "X_test_frames.dat"
    }

    if not META_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo de metadata en {META_PATH}")

    meta = np.load(META_PATH, allow_pickle=True)

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
        "LAMBDA_PHYS": args.lambda_phys,
        "LAMBDA_TEMP": args.lambda_temp,
        "LAMBDA_ENERGY": args.lambda_energy,
        "GAMMA": args.gamma,
        "MODEL": "PI-DeepONet_CORAL_Ordinal_Energy",
        "DEVICE": str(DEVICE)
    }

    logger.start_experiment(hyperparameters=hyperparameters, data_files=data_files)

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
    best_model_path = CHECKPOINT_DIR / f"pideeponet_coral_lphys_{args.lambda_phys}_ltemp_{args.lambda_temp}_lenergy_{args.lambda_energy}_seed_{args.seed}_best.pth"

    print("Iniciando entrenamiento CORAL con Restricción de Energía y Temporal...")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_loss_temp, tr_loss_energy = train_epoch(
            model, train_loader, optimizer, DEVICE, 
            args.lambda_phys, args.lambda_temp, args.lambda_energy, args.gamma, class_weights_tensor
        )
        val_loss, val_acc, val_loss_temp, val_loss_energy = evaluate(
            model, val_loader, DEVICE, 
            args.lambda_phys, args.lambda_temp, args.lambda_energy, args.gamma, class_weights_tensor
        )

        scheduler.step(val_loss)

        extra_metrics = {
            "tr_loss_temp": round(tr_loss_temp, 6),
            "val_loss_temp": round(val_loss_temp, 6),
            "tr_loss_energy": round(tr_loss_energy, 6),
            "val_loss_energy": round(val_loss_energy, 6)
        }
        logger.log_epoch(epoch, tr_loss, tr_acc, val_loss, val_acc, extra_metrics=extra_metrics)

        print(f"Época [{epoch:02d}/{EPOCHS:02d}] | Train Loss: {tr_loss:.4f} (Temp: {tr_loss_temp:.6f}, Energy: {tr_loss_energy:.6f}) - Acc: {tr_acc:.2f}% | Val Loss: {val_loss:.4f} (Temp: {val_loss_temp:.6f}, Energy: {val_loss_energy:.6f}) - Acc: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    print("\nCargando mejor modelo CORAL + Energía para evaluación Cross-Domain...")
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, _, _ = evaluate(
        model, test_loader, DEVICE, 
        args.lambda_phys, args.lambda_temp, args.lambda_energy, args.gamma, class_weights_tensor
    )

    test_metrics = {
        "Target_Domain": str(test_sets[0]),
        "Loss": float(test_loss),
        "Accuracy_Percent": float(test_acc)
    }

    logger.end_experiment(test_metrics=test_metrics)

    run_post_hoc_diagnostics(best_model_path, model_type="pideeponet", output_dir=logger.output_dir, num_classes=num_classes)

if __name__ == "__main__":
    main()

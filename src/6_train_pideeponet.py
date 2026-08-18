import argparse
import os
from pathlib import Path
import random
import time
from evaluate_diagnostics import run_post_hoc_diagnostics

from experiment_logger import ExperimentLogger
from m5b_pideeponet_model import MemmapCSIDataset, PIDeepONet, compute_pi_loss
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    lambda_energy,
    lambda_mono,
    margin,
    mix_ratio,
    gamma,
):
  model.train()
  running_loss, running_loss_energy, running_loss_mono = 0.0, 0.0, 0.0
  correct, total = 0, 0

  for x_b, y_people in dataloader:
    x_b, y_people = x_b.to(device), y_people.to(device)

    optimizer.zero_grad()
    logits, latent_field = model(x_b)

    loss, _, loss_energy, loss_mono = compute_pi_loss(
        logits,
        latent_field,
        y_people,
        lambda_energy=lambda_energy,
        gamma=gamma,
        lambda_mono=lambda_mono,
        margin=margin,
        mix_ratio=mix_ratio,
    )
    loss.backward()
    optimizer.step()

    running_loss += loss.item() * x_b.size(0)
    running_loss_energy += loss_energy.item() * x_b.size(0)
    running_loss_mono += loss_mono.item() * x_b.size(0)

    probs = torch.sigmoid(logits)
    preds = torch.sum(probs > 0.5, dim=1)

    correct += (preds == y_people).sum().item()
    total += y_people.size(0)

  return (
      running_loss / total,
      (correct / total) * 100.0,
      running_loss_energy / total,
      running_loss_mono / total,
  )


def evaluate(
    model,
    dataloader,
    device,
    lambda_energy,
    lambda_mono,
    margin,
    mix_ratio,
    gamma,
):
  model.eval()
  running_loss, running_loss_energy, running_loss_mono = 0.0, 0.0, 0.0
  correct, total = 0, 0

  with torch.no_grad():
    for x_b, y_people in dataloader:
      x_b, y_people = x_b.to(device), y_people.to(device)

      logits, latent_field = model(x_b)
      loss, _, loss_energy, loss_mono = compute_pi_loss(
          logits,
          latent_field,
          y_people,
          lambda_energy=lambda_energy,
          gamma=gamma,
          lambda_mono=lambda_mono,
          margin=margin,
          mix_ratio=mix_ratio,
      )

      running_loss += loss.item() * x_b.size(0)
      running_loss_energy += loss_energy.item() * x_b.size(0)
      running_loss_mono += loss_mono.item() * x_b.size(0)

      probs = torch.sigmoid(logits)
      preds = torch.sum(probs > 0.5, dim=1)

      correct += (preds == y_people).sum().item()
      total += y_people.size(0)

  return (
      running_loss / total,
      (correct / total) * 100.0,
      running_loss_energy / total,
      running_loss_mono / total,
  )


def main():
  parser = argparse.ArgumentParser(
      description=(
          "Entrenamiento PI-DeepONet CORAL + Monotonicidad Suave (5 Clases)"
      )
  )
  parser.add_argument(
      "--seed", type=int, default=42, help="Semilla aleatoria"
  )
  parser.add_argument(
      "--lambda_energy",
      type=float,
      default=0.01,
      help="Coeficiente para regularización de energía RCS",
  )
  parser.add_argument(
      "--lambda_mono",
      type=float,
      default=0.001,
      help="Coeficiente para monotonicidad suave",
  )
  parser.add_argument(
      "--margin",
      type=float,
      default=0.0,
      help="Margen de tolerancia para monotonicidad",
  )
  parser.add_argument(
      "--warmup_epochs",
      type=int,
      default=5,
      help="Épocas de warm-up sin monotonicidad",
  )
  parser.add_argument(
      "--mix_ratio",
      type=float,
      default=0.0,
      help="Proporción de pares aleatorios (0.0=vecinos, 0.3=70/30)",
  )
  parser.add_argument(
      "--gamma",
      type=float,
      default=1.0,
      help="Factor de escala para la energía",
  )
  args = parser.parse_args()

  set_seed(args.seed)

  print("=== PI-DEEPONET CORAL + MONOTONICIDAD SUAVE (5 CLASES) ===")
  print(
      f"Dispositivo activo: {DEVICE} ({torch.cuda.get_device_name(0)}) |"
      f" Semilla: {args.seed}"
  )
  print(
      f"Parámetros -> L_energy: {args.lambda_energy} | L_mono: {args.lambda_mono}"
      f" | Margin: {args.margin} | Warmup: {args.warmup_epochs} | MixRatio:"
      f" {args.mix_ratio}"
  )

  exp_name = (
      f"PI_DeepONet_CORAL_Lenergy_{args.lambda_energy}_Lmono_{args.lambda_mono}_Margin_{args.margin}_Mix_{args.mix_ratio}_Seed_{args.seed}"
  )
  logger = ExperimentLogger(
      experiment_name=exp_name, base_dir=PROJECT_DIR / "runs"
  )

  data_files = {
      "metadata": META_PATH,
      "X_train": TENSORS_DIR / "X_train_frames.dat",
      "X_val": TENSORS_DIR / "X_val_frames.dat",
      "X_test": TENSORS_DIR / "X_test_frames.dat",
  }

  meta = np.load(META_PATH, allow_pickle=True)
  train_y, val_y, test_y = (
      meta["train_y_people"],
      meta["val_y_people"],
      meta["test_y_people"],
  )
  num_classes = len(np.unique(train_y))

  hyperparameters = {
      "SEED": args.seed,
      "NUM_CLASSES": num_classes,
      "BATCH_SIZE": BATCH_SIZE,
      "EPOCHS": EPOCHS,
      "LEARNING_RATE": LEARNING_RATE,
      "LAMBDA_ENERGY": args.lambda_energy,
      "LAMBDA_MONO": args.lambda_mono,
      "MARGIN": args.margin,
      "WARMUP_EPOCHS": args.warmup_epochs,
      "MIX_RATIO": args.mix_ratio,
      "GAMMA": args.gamma,
      "MODEL": "PI-DeepONet_CORAL_Monotonic_Energy",
      "DEVICE": str(DEVICE),
  }

  logger.start_experiment(
      hyperparameters=hyperparameters, data_files=data_files
  )

  class_weights = compute_class_weight(
      class_weight="balanced", classes=np.unique(train_y), y=train_y
  )
  class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(
      DEVICE
  )

  test_sets = np.unique(meta["test_set_ids"])

  train_dataset = MemmapCSIDataset(
      TENSORS_DIR / "X_train_frames.dat",
      meta["tr_shape"],
      meta["train_starts"],
      train_y,
  )
  val_dataset = MemmapCSIDataset(
      TENSORS_DIR / "X_val_frames.dat",
      meta["va_shape"],
      meta["val_starts"],
      val_y,
  )
  test_dataset = MemmapCSIDataset(
      TENSORS_DIR / "X_test_frames.dat",
      meta["te_shape"],
      meta["test_starts"],
      test_y,
  )

  train_loader = DataLoader(
      train_dataset,
      batch_size=BATCH_SIZE,
      shuffle=True,
      num_workers=NUM_WORKERS,
      pin_memory=True,
  )
  val_loader = DataLoader(
      val_dataset,
      batch_size=BATCH_SIZE,
      shuffle=False,
      num_workers=NUM_WORKERS,
      pin_memory=True,
  )
  test_loader = DataLoader(
      test_dataset,
      batch_size=BATCH_SIZE,
      shuffle=False,
      num_workers=NUM_WORKERS,
      pin_memory=True,
  )

  model = PIDeepONet(num_classes=num_classes, latent_dim=128).to(DEVICE)
  optimizer = torch.optim.AdamW(
      model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
  )
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, mode="min", factor=0.5, patience=2
  )

  best_val_loss = float("inf")
  best_model_path = (
      CHECKPOINT_DIR
      / f"pideeponet_coral_lmono_{args.lambda_mono}_seed_{args.seed}_best.pth"
  )

  for epoch in range(1, EPOCHS + 1):
    active_lambda_mono = (
        args.lambda_mono if epoch > args.warmup_epochs else 0.0
    )

    tr_loss, tr_acc, tr_energy, tr_mono = train_epoch(
        model,
        train_loader,
        optimizer,
        DEVICE,
        args.lambda_energy,
        active_lambda_mono,
        args.margin,
        args.mix_ratio,
        args.gamma,
    )
    val_loss, val_acc, val_energy, val_mono = evaluate(
        model,
        val_loader,
        DEVICE,
        args.lambda_energy,
        active_lambda_mono,
        args.margin,
        args.mix_ratio,
        args.gamma,
    )

    scheduler.step(val_loss)

    extra_metrics = {
        "tr_loss_energy": round(tr_energy, 6),
        "val_loss_energy": round(val_energy, 6),
        "tr_loss_mono": round(tr_mono, 6),
        "val_loss_mono": round(val_mono, 6),
    }
    logger.log_epoch(
        epoch, tr_loss, tr_acc, val_loss, val_acc, extra_metrics=extra_metrics
    )

    print(
        f"Época [{epoch:02d}/{EPOCHS:02d}] | Train Loss: {tr_loss:.4f} (Energy:"
        f" {tr_energy:.6f}, Mono: {tr_mono:.6f}) - Acc: {tr_acc:.2f}% | Val"
        f" Loss: {val_loss:.4f} (Energy: {val_energy:.6f}, Mono:"
        f" {val_mono:.6f}) - Acc: {val_acc:.2f}%"
    )

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      torch.save(model.state_dict(), best_model_path)

  print("\nCargando mejor modelo para evaluación Cross-Domain...")
  model.load_state_dict(torch.load(best_model_path))
  test_loss, test_acc, _, _ = evaluate(
      model,
      test_loader,
      DEVICE,
      args.lambda_energy,
      args.lambda_mono,
      args.margin,
      args.mix_ratio,
      args.gamma,
  )

  test_metrics = {
      "Target_Domain": str(test_sets[0]),
      "Loss": float(test_loss),
      "Accuracy_Percent": float(test_acc),
  }
  logger.end_experiment(test_metrics=test_metrics)

  run_post_hoc_diagnostics(
      best_model_path,
      model_type="pideeponet",
      output_dir=logger.output_dir,
      num_classes=num_classes,
  )


if __name__ == "__main__":
  main()

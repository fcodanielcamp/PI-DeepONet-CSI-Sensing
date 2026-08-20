import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from m5b_pideeponet_model import MemmapCSIDataset, PIDeepONet

# ==============================================================================
# DIAGNÓSTICO: ¿SE AMPLIFICA EL DESPLAZAMIENTO DE DOMINIO EN LA CAPA fc
# (b_out), QUE NO TIENE NINGUNA NORMALIZACIÓN DESPUÉS DE ELLA?
# ==============================================================================
# Diagnóstico A (BatchNorm, capas convolucionales) mostró desplazamientos
# casi nulos (|z_shift| ~0.05-0.11). Este script repite el mismo tipo de
# análisis pero sobre b_out -- el vector de 128 dimensiones que produce
# BranchNet.fc y que alimenta directamente al clasificador CORAL, al decoder
# físico, y a loss_energy. Si aquí el desplazamiento es mucho mayor que en
# las capas convolucionales, confirma que la capa fc sin normalizar es el
# punto donde se amplifica el domain-shift.
# ==============================================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
TENSORS_DIR = DATA_DIR / "processed_tensors"
META_PATH = TENSORS_DIR / "dataset_mc1_rx1_80mhz_meta.npz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_SIZE = 8192  # ventanas por dominio, suficiente para estadísticas estables por dimensión


def get_sample_loader(dataset, n, batch_size, seed):
  rng = np.random.default_rng(seed)
  n = min(n, len(dataset))
  idx = rng.choice(len(dataset), size=n, replace=False)
  subset = Subset(dataset, idx.tolist())
  return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)


def collect_bout(model, loader, device):
  """Corre el modelo y recolecta b_out (128-d) junto con la clase real de
  cada muestra. b_out ya es devuelto directamente por PIDeepONet.forward,
  no requiere hooks."""
  model.eval()
  all_y = []
  all_bout = []

  with torch.no_grad():
    for x_b, y_b in loader:
      x_b = x_b.to(device)
      _, _, b_out, _ = model(x_b)
      all_y.append(y_b)
      all_bout.append(b_out.cpu())

  return torch.cat(all_y).numpy(), torch.cat(all_bout).numpy()


def report_dimension_shift(y_train, bout_train, y_test, bout_test):
  print("\n=== DIAGNÓSTICO C: desplazamiento por dimensión en b_out (salida de fc, sin normalizar) ===")

  train_mean = bout_train.mean(axis=0)   # (128,)
  train_std = bout_train.std(axis=0) + 1e-8
  test_mean = bout_test.mean(axis=0)

  z_shift = (test_mean - train_mean) / train_std

  print(f"\n  Dimensiones totales de b_out: {len(train_mean)}")
  print(f"  |z_shift| medio entre dominios:  {np.abs(z_shift).mean():.3f}")
  print(f"  |z_shift| mediana:               {np.median(np.abs(z_shift)):.3f}")
  print(f"  |z_shift| máximo (peor dim.):    {np.abs(z_shift).max():.3f}")
  print(f"  Dimensiones con |z_shift| > 1:    {(np.abs(z_shift) > 1).sum()} / {len(z_shift)}")
  print(f"  Dimensiones con |z_shift| > 2:    {(np.abs(z_shift) > 2).sum()} / {len(z_shift)}")
  print(f"  Dimensiones con |z_shift| > 3:    {(np.abs(z_shift) > 3).sum()} / {len(z_shift)}")

  top_k = np.argsort(-np.abs(z_shift))[:5]
  print("\n  Top-5 dimensiones más desplazadas:")
  for d in top_k:
    print(f"    dim {d:3d}: train_mean={train_mean[d]:+.3f}  test_mean={test_mean[d]:+.3f}  z_shift={z_shift[d]:+.3f}")

  return z_shift


def report_norm_by_class(name, y, bout, num_classes):
  norms = np.linalg.norm(bout, axis=1)
  print(f"\n--- {name}: ||b_out|| por clase real ---")
  print(f"  {'Clase':<8}{'N':<10}{'||b_out|| media':<20}{'||b_out|| std':<18}")
  for c in range(num_classes):
    mask = y == c
    n = mask.sum()
    if n == 0:
      print(f"  {c:<8}{0:<10}(sin muestras)")
      continue
    print(f"  {c:<8}{n:<10}{norms[mask].mean():<20.4f}{norms[mask].std():<18.4f}")


def main():
  parser = argparse.ArgumentParser(
      description="Diagnóstico de desplazamiento de dominio en b_out (salida de fc sin normalizar)"
  )
  parser.add_argument("--model_path", type=str, required=True)
  parser.add_argument("--sample_size", type=int, default=SAMPLE_SIZE)
  args = parser.parse_args()

  meta = np.load(META_PATH, allow_pickle=True)
  train_y = meta["train_y_people"]
  test_y = meta["test_y_people"]
  num_classes = len(np.unique(train_y))

  train_dataset = MemmapCSIDataset(
      TENSORS_DIR / "X_train_frames.dat", meta["tr_shape"], meta["train_starts"], train_y
  )
  test_dataset = MemmapCSIDataset(
      TENSORS_DIR / "X_test_frames.dat", meta["te_shape"], meta["test_starts"], test_y
  )

  train_loader = get_sample_loader(train_dataset, n=args.sample_size, batch_size=512, seed=0)
  test_loader = get_sample_loader(test_dataset, n=args.sample_size, batch_size=512, seed=1)

  model = PIDeepONet(num_classes=num_classes, latent_dim=128).to(DEVICE)
  model.load_state_dict(torch.load(args.model_path, map_location=DEVICE, weights_only=True))
  model.eval()

  print(f"Modelo cargado desde: {args.model_path}")
  print(f"Muestra por dominio: {args.sample_size} ventanas")

  y_train, bout_train = collect_bout(model, train_loader, DEVICE)
  y_test, bout_test = collect_bout(model, test_loader, DEVICE)

  report_dimension_shift(y_train, bout_train, y_test, bout_test)
  report_norm_by_class("TRAIN", y_train, bout_train, num_classes)
  report_norm_by_class("TEST", y_test, bout_test, num_classes)

  print("\n=== COMPARACIÓN DE REFERENCIA ===")
  print("  Diagnóstico A (capas convolucionales, con BatchNorm): |z_shift| medio ~0.05-0.11")
  print("  Diagnóstico C (b_out, capa fc SIN normalizar):        ver arriba")
  print("  Si el valor de arriba es sustancialmente mayor (p.ej. >0.5-1.0), confirma")
  print("  que la capa fc sin normalizar amplifica el desplazamiento de dominio.")


if __name__ == "__main__":
  main()

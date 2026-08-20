import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from m5b_pideeponet_model import MemmapCSIDataset, PIDeepONet

# ==============================================================================
# DIAGNÓSTICO DEL CABEZAL DE CLASIFICACIÓN: ¿HAY SEÑAL, O SOLO MALA CALIBRACIÓN?
# ==============================================================================
# Descartado el desplazamiento de BatchNorm (Diagnóstico A/B), este script
# examina el escalar CORAL proyectado (antes de sumar coral_bias) y la norma
# del campo latente, condicionados por la clase REAL de personas, en train y
# en test. Responde una pregunta concreta:
#
#   ¿El escalar proyectado / la norma del latente SIGUE variando de forma
#   coherente con el número real de personas en el dominio de test (aunque
#   esté desplazado respecto a los umbrales aprendidos en train), o se vuelve
#   plano / sin relación con la ocupación real?
#
# Caso 1 (varía, pero desplazado) -> problema de CALIBRACIÓN del cabezal.
# Caso 2 (plano, sin relación)    -> problema de GENERALIZACIÓN de la
#                                     representación (Branch/Trunk), no
#                                     arreglable ajustando solo el cabezal.
# ==============================================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
TENSORS_DIR = DATA_DIR / "processed_tensors"
META_PATH = TENSORS_DIR / "dataset_mc1_rx1_80mhz_meta.npz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_SAMPLE_SIZE = 20000  # suficiente para percentiles estables por clase, barato de computar


def get_sample_loader(dataset, n, batch_size, seed):
  rng = np.random.default_rng(seed)
  n = min(n, len(dataset))
  idx = rng.choice(len(dataset), size=n, replace=False)
  subset = Subset(dataset, idx.tolist())
  return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)


def collect_head_signals(model, loader, device):
  """Corre el modelo sobre un loader y recolecta, por muestra: la clase
  real, el escalar CORAL proyectado (antes de coral_bias) y la norma L2 del
  campo latente (la misma cantidad que usa loss_energy)."""
  model.eval()
  projected_hook_out = []

  def hook(module, inp, out):
    projected_hook_out.append(out.detach())

  h = model.classifier.register_forward_hook(hook)

  all_y = []
  all_projected = []
  all_latent_norm = []

  with torch.no_grad():
    for x_b, y_b in loader:
      x_b = x_b.to(device)
      projected_hook_out.clear()

      logits, latent_field, b_out, H_hat = model(x_b)

      projected = projected_hook_out[0].squeeze(1).cpu()  # (B,)
      latent_norm = torch.norm(latent_field, p=2, dim=1).cpu()  # (B,)

      all_y.append(y_b)
      all_projected.append(projected)
      all_latent_norm.append(latent_norm)

  h.remove()

  return (
      torch.cat(all_y).numpy(),
      torch.cat(all_projected).numpy(),
      torch.cat(all_latent_norm).numpy(),
  )


def report_by_class(name, y, projected, latent_norm, num_classes):
  print(f"\n--- {name} ---")
  print(f"  {'Clase':<8}{'N':<10}{'projected (media)':<20}{'projected (std)':<18}{'latent_norm (media)':<20}")
  for c in range(num_classes):
    mask = y == c
    n = mask.sum()
    if n == 0:
      print(f"  {c:<8}{0:<10}{'(sin muestras)':<20}")
      continue
    print(
        f"  {c:<8}{n:<10}{projected[mask].mean():<20.4f}"
        f"{projected[mask].std():<18.4f}{latent_norm[mask].mean():<20.4f}"
    )


def main():
  parser = argparse.ArgumentParser(
      description="Diagnóstico del cabezal CORAL: señal vs. calibración entre dominios"
  )
  parser.add_argument("--model_path", type=str, required=True)
  parser.add_argument("--train_sample_size", type=int, default=TRAIN_SAMPLE_SIZE)
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

  train_loader = get_sample_loader(train_dataset, n=args.train_sample_size, batch_size=512, seed=0)
  test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)

  model = PIDeepONet(num_classes=num_classes, latent_dim=128).to(DEVICE)
  model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
  model.eval()

  coral_bias = model.coral_bias.detach().cpu().numpy()
  print(f"Modelo cargado desde: {args.model_path}")
  print(f"Umbrales coral_bias aprendidos: {coral_bias}")

  y_train, proj_train, norm_train = collect_head_signals(model, train_loader, DEVICE)
  y_test, proj_test, norm_test = collect_head_signals(model, test_loader, DEVICE)

  report_by_class("TRAIN -- escalar proyectado y norma latente por clase real", y_train, proj_train, norm_train, num_classes)
  report_by_class("TEST  -- escalar proyectado y norma latente por clase real", y_test, proj_test, norm_test, num_classes)

  print("\n=== INTERPRETACIÓN ===")
  # Correlación simple entre clase real y escalar proyectado, por dominio
  corr_train = np.corrcoef(y_train, proj_train)[0, 1]
  corr_test = np.corrcoef(y_test, proj_test)[0, 1]
  print(f"  Correlación (clase real, projected) en TRAIN: {corr_train:.3f}")
  print(f"  Correlación (clase real, projected) en TEST:  {corr_test:.3f}")

  # Cuántos umbrales supera, en promedio, el escalar proyectado de TEST
  mean_proj_test = proj_test.mean()
  thresholds_exceeded = (mean_proj_test + coral_bias > 0).sum()
  print(f"\n  Escalar proyectado medio en TEST: {mean_proj_test:.4f}")
  print(f"  Umbrales superados en promedio (de {len(coral_bias)}): {thresholds_exceeded}")

  if abs(corr_test) > 0.3:
    print("\n  -> SÍ hay señal relacionada con la ocupación real en el dominio de test,")
    print("     pero está desplazada/mal calibrada respecto a los umbrales de train.")
    print("     Esto apunta a un problema de CALIBRACIÓN del cabezal, no de representación.")
  else:
    print("\n  -> NO hay relación apreciable entre el escalar proyectado y la ocupación")
    print("     real en el dominio de test. La representación (Branch/Trunk) no está")
    print("     generalizando información relevante para el conteo a este dominio nuevo.")


if __name__ == "__main__":
  main()

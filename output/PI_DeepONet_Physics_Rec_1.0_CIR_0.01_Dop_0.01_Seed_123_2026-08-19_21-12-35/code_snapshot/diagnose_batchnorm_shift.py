import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from m5b_pideeponet_model import MemmapCSIDataset, PIDeepONet

# ==============================================================================
# DIAGNÓSTICO DE DESPLAZAMIENTO DE BATCHNORM ENTRE DOMINIOS (TRAIN vs TEST)
# ==============================================================================
# No reentrena nada. Reutiliza un checkpoint ya entrenado y responde dos
# preguntas con evidencia directa:
#
#   DIAGNÓSTICO A (correlacional): ¿las activaciones reales del dominio de
#   test están desplazadas respecto a las estadísticas congeladas
#   (running_mean/running_var) que BatchNorm aprendió sobre el dominio de
#   entrenamiento?
#
#   DIAGNÓSTICO B (causal, técnica AdaBN — Li et al. 2018): si, en vez de usar
#   esas estadísticas congeladas, se le permite a BatchNorm usar las
#   estadísticas del propio batch de test en el momento de la inferencia,
#   ¿mejora sustancialmente el colapso hacia una sola clase?
# ==============================================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
TENSORS_DIR = DATA_DIR / "processed_tensors"
META_PATH = TENSORS_DIR / "dataset_mc1_rx1_80mhz_meta.npz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_SIZE = 4096  # ventanas por dominio para el Diagnóstico A; suficiente para estadísticas estables y barato de calcular


def get_sample_loader(dataset, n, batch_size, seed):
  rng = np.random.default_rng(seed)
  n = min(n, len(dataset))
  idx = rng.choice(len(dataset), size=n, replace=False)
  subset = Subset(dataset, idx.tolist())
  return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)


def collect_prebn_stats(model, loader, device):
  """Registra hooks en bn1/bn2/bn3 de BranchNet y devuelve, por capa, la
  media y varianza REALES (por canal) de las activaciones justo ANTES de
  BatchNorm, calculadas sobre el dominio que traiga el loader."""
  branch = model.branch
  bn_layers = {"bn1": branch.bn1, "bn2": branch.bn2, "bn3": branch.bn3}

  activations = {name: [] for name in bn_layers}
  hooks = []

  def make_hook(name):
    def hook(module, inp):
      # inp es una tupla; inp[0] es el tensor que entra a BatchNorm2d, (B,C,H,W)
      activations[name].append(inp[0].detach())
    return hook

  for name, layer in bn_layers.items():
    hooks.append(layer.register_forward_pre_hook(make_hook(name)))

  model.eval()
  with torch.no_grad():
    for x_b, _ in loader:
      x_b = x_b.to(device)
      model(x_b)

  for h in hooks:
    h.remove()

  stats = {}
  for name in bn_layers:
    all_acts = torch.cat(activations[name], dim=0)
    mean = all_acts.mean(dim=(0, 2, 3))
    var = all_acts.var(dim=(0, 2, 3), unbiased=False)
    stats[name] = (mean.cpu(), var.cpu())
  return stats


def report_shift(model, train_stats, test_stats):
  print("\n=== DIAGNÓSTICO A: desplazamiento de activaciones pre-BatchNorm (train vs test) ===")
  branch = model.branch
  bn_layers = {"bn1": branch.bn1, "bn2": branch.bn2, "bn3": branch.bn3}

  for name, layer in bn_layers.items():
    running_mean = layer.running_mean.cpu()
    running_var = layer.running_var.cpu()
    eps = layer.eps

    test_mean, test_var = test_stats[name]
    train_mean, _ = train_stats[name]

    # Desplazamiento respecto a las estadísticas congeladas que BatchNorm usa
    # en inferencia, expresado en unidades de desviación estándar del propio
    # dominio de entrenamiento (running_std)
    z_shift_test = (test_mean - running_mean) / torch.sqrt(running_var + eps)
    var_ratio = test_var / (running_var + eps)

    # Referencia: el mismo cálculo pero sobre una muestra de TRAIN.
    # Como running_mean/var se ajustaron sobre datos de train, esto debería
    # dar un desplazamiento cercano a 0 -- es el "control" del experimento.
    z_shift_train = (train_mean - running_mean) / torch.sqrt(running_var + eps)

    print(f"\n--- {name} ({len(running_mean)} canales) ---")
    print(f"  |z_shift| medio en TRAIN (control, debería ser ~0): {z_shift_train.abs().mean():.3f}")
    print(f"  |z_shift| medio en TEST  (dominio no visto):        {z_shift_test.abs().mean():.3f}")
    print(f"  |z_shift| máximo en TEST (peor canal):              {z_shift_test.abs().max():.3f}")
    print(f"  Razón de varianza test/running (media):             {var_ratio.mean():.3f}")


def adabn_diagnostic(model, test_loader, device, num_classes):
  """Diagnóstico causal: ¿mejora el colapso si BatchNorm usa las
  estadísticas del propio batch de TEST, en vez de running_mean/var
  congeladas de train? Técnica AdaBN, sin reentrenar nada."""
  print("\n=== DIAGNÓSTICO B: AdaBN -- BatchNorm con estadísticas del batch de TEST ===")

  model_adabn = copy.deepcopy(model)
  model_adabn.eval()
  # Solo las capas BatchNorm se fuerzan a modo train (usan estadística del
  # batch actual en vez de running_mean/var). El resto del modelo (Dropout,
  # etc.) permanece en modo eval.
  for m in model_adabn.modules():
    if isinstance(m, nn.BatchNorm2d):
      m.train()

  correct_frozen, correct_adabn, total = 0, 0, 0
  pred_counts_frozen = np.zeros(num_classes, dtype=int)
  pred_counts_adabn = np.zeros(num_classes, dtype=int)

  with torch.no_grad():
    for x_b, y_b in test_loader:
      x_b, y_b = x_b.to(device), y_b.to(device)

      logits_frozen, _, _, _ = model(x_b)
      preds_frozen = torch.sum(torch.sigmoid(logits_frozen) > 0.5, dim=1)

      logits_adabn, _, _, _ = model_adabn(x_b)
      preds_adabn = torch.sum(torch.sigmoid(logits_adabn) > 0.5, dim=1)

      correct_frozen += (preds_frozen == y_b).sum().item()
      correct_adabn += (preds_adabn == y_b).sum().item()
      total += y_b.size(0)

      for c in range(num_classes):
        pred_counts_frozen[c] += (preds_frozen == c).sum().item()
        pred_counts_adabn[c] += (preds_adabn == c).sum().item()

  acc_frozen = 100.0 * correct_frozen / total
  acc_adabn = 100.0 * correct_adabn / total
  delta = acc_adabn - acc_frozen

  print(f"\n  Accuracy TEST con BatchNorm congelado (comportamiento actual): {acc_frozen:.2f}%")
  print(f"  Distribución de predicciones (congelado): {pred_counts_frozen.tolist()}")
  print(f"\n  Accuracy TEST con estadísticas de batch de TEST (AdaBN):        {acc_adabn:.2f}%")
  print(f"  Distribución de predicciones (AdaBN):     {pred_counts_adabn.tolist()}")
  print(f"\n  Delta de accuracy (AdaBN - congelado): {delta:+.2f} puntos porcentuales")

  if delta > 5:
    print("  -> Evidencia CAUSAL de que el desplazamiento de estadísticas de")
    print("     BatchNorm contribuye significativamente al colapso hacia una sola clase.")
  else:
    print("  -> El desplazamiento de BatchNorm NO explica por sí solo el colapso;")
    print("     hay que buscar la causa en otro componente (p.ej. escala de coral_bias,")
    print("     o la estructura ordinal de CORAL bajo domain-shift).")


def main():
  parser = argparse.ArgumentParser(
      description="Diagnóstico de desplazamiento de BatchNorm entre train y test (sin reentrenar)"
  )
  parser.add_argument(
      "--model_path", type=str, required=True,
      help="Ruta al checkpoint .pth ya entrenado (p.ej. models/pideeponet_physics_seed_42_best.pth)"
  )
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

  train_loader_sample = get_sample_loader(train_dataset, n=args.sample_size, batch_size=512, seed=0)
  test_loader_sample = get_sample_loader(test_dataset, n=args.sample_size, batch_size=512, seed=1)
  test_loader_full = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)

  model = PIDeepONet(num_classes=num_classes, latent_dim=128).to(DEVICE)
  model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
  model.eval()

  print(f"Modelo cargado desde: {args.model_path}")
  print(f"Muestra por dominio para el Diagnóstico A: {args.sample_size} ventanas")

  train_stats = collect_prebn_stats(model, train_loader_sample, DEVICE)
  test_stats = collect_prebn_stats(model, test_loader_sample, DEVICE)
  report_shift(model, train_stats, test_stats)

  adabn_diagnostic(model, test_loader_full, DEVICE, num_classes)


if __name__ == "__main__":
  main()

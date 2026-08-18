import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TENSORS_DIR = DATA_DIR / "processed_tensors"
META_PATH = TENSORS_DIR / "dataset_mc1_rx1_80mhz_meta.npz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MemmapCSIDataset(Dataset):

  def __init__(self, dat_path, shape, starts, y_people, window_size=25):
    self.dat_path = str(dat_path)
    self.shape = tuple(shape)
    self.starts = starts
    self.y_people = torch.tensor(y_people, dtype=torch.long)
    self.window_size = window_size
    self.memmap = None

  def _init_memmap(self):
    if self.memmap is None:
      self.memmap = np.memmap(
          self.dat_path, dtype='float16', mode='r', shape=self.shape
      )

  def __len__(self):
    return len(self.starts)

  def __getitem__(self, idx):
    self._init_memmap()
    st = self.starts[idx]
    win = self.memmap[st : st + self.window_size, :, :].astype(np.float32)
    # Forma retenida: (2, 25, 241) -> (Canales, Tiempo/Frames, Subportadoras)
    x_tensor = torch.from_numpy(win).permute(2, 0, 1).contiguous()
    return x_tensor, self.y_people[idx]


class BranchNet(nn.Module):

  def __init__(self, latent_dim=128):
    super(BranchNet, self).__init__()
    self.conv1 = nn.Conv2d(
        in_channels=2, out_channels=16, kernel_size=(3, 5), padding=(1, 2)
    )
    self.bn1 = nn.BatchNorm2d(16)
    self.conv2 = nn.Conv2d(
        in_channels=16, out_channels=32, kernel_size=(3, 5), padding=(1, 2)
    )
    self.bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(
        in_channels=32, out_channels=64, kernel_size=(3, 5), padding=(1, 2)
    )
    self.bn3 = nn.BatchNorm2d(64)
    self.pool = nn.MaxPool2d(kernel_size=(2, 2))
    self.act = nn.Mish()
    self.fc = nn.Linear(64 * 3 * 30, latent_dim)

  def forward(self, x):
    x = self.pool(self.act(self.bn1(self.conv1(x))))
    x = self.pool(self.act(self.bn2(self.conv2(x))))
    x = self.pool(self.act(self.bn3(self.conv3(x))))
    x = x.view(x.size(0), -1)
    return self.fc(x)


class TrunkNet(nn.Module):

  def __init__(self, latent_dim=128):
    super(TrunkNet, self).__init__()
    self.net = nn.Sequential(
        nn.Linear(2, 64),
        nn.Mish(),
        nn.Linear(64, 128),
        nn.Mish(),
        nn.Linear(128, latent_dim),
        nn.Mish(),
    )

  def forward(self, grid_2d):
    return self.net(grid_2d)


class PhysicalDecoder(nn.Module):
  """Decoder que proyecta el campo latente z = [z_h || z_D] hacia el espectro reconstituido (B, 2, 25, 241)"""

  def __init__(self, latent_dim=128, time_steps=25, subcarriers=241):
    super(PhysicalDecoder, self).__init__()
    self.time_steps = time_steps
    self.subcarriers = subcarriers

    self.fc = nn.Sequential(
        nn.Linear(latent_dim, 256),
        nn.Mish(),
        nn.Linear(256, 512),
        nn.Mish(),
        nn.Linear(512, 2 * time_steps * subcarriers),
    )

  def forward(self, z):
    B = z.size(0)
    out = self.fc(z)
    return out.view(B, 2, self.time_steps, self.subcarriers)


class PIDeepONet(nn.Module):

  def __init__(self, num_classes=5, latent_dim=128):
    super(PIDeepONet, self).__init__()
    self.num_classes = num_classes
    self.latent_dim = latent_dim
    # z se divide en z_h (64) y z_D (64)
    self.latent_h_dim = latent_dim // 2
    self.latent_D_dim = latent_dim // 2

    self.branch = BranchNet(latent_dim=latent_dim)
    self.trunk = TrunkNet(latent_dim=latent_dim)

    # Rama decodificadora física (para reconstrucción supervisada/consistencia)
    self.decoder = PhysicalDecoder(
        latent_dim=latent_dim, time_steps=25, subcarriers=241
    )

    self.classifier = nn.Sequential(
        nn.Linear(latent_dim, 64),
        nn.Mish(),
        nn.Dropout(0.2),
        nn.Linear(64, 1, bias=False),
    )

    # Inicialización de umbrales ordinales decrecientes para CORAL
    num_thresholds = num_classes - 1
    initial_biases = torch.linspace(2.0, -2.0, num_thresholds)
    self.coral_bias = nn.Parameter(initial_biases)

    t_coords = np.linspace(0, 1, 25, dtype=np.float32)
    s_coords = np.linspace(0, 1, 241, dtype=np.float32)
    grid_t, grid_s = np.meshgrid(t_coords, s_coords, indexing='ij')
    grid_2d = np.stack([grid_t.ravel(), grid_s.ravel()], axis=-1)

    self.register_buffer('grid_2d', torch.from_numpy(grid_2d))

  def forward(self, x_branch):
    b_out = self.branch(x_branch)  # Shape: (B, 128)
    t_out = self.trunk(self.grid_2d)  # Shape: (6025, 128)

    latent_field = torch.matmul(b_out, t_out.T)  # Shape: (B, 6025)
    field_summary = b_out * torch.mean(t_out, dim=0, keepdim=True)

    projected = self.classifier(field_summary)
    logits = projected + self.coral_bias

    # Reconstrucción física a partir del latente interpretado
    H_hat = self.decoder(b_out)  # Shape: (B, 2, 25, 241)

    return logits, latent_field, b_out, H_hat


def compute_pi_loss(
    logits,
    latent_field,
    b_out,
    H_hat,
    x_obs,
    y_people,
    lambda_energy=0.01,
    gamma=1.0,
    lambda_mono=0.0,
    margin=0.0,
    mix_ratio=0.0,
    lambda_rec=0.1,
    lambda_cir=0.01,
    lambda_dop=0.01,
):
  device = logits.device
  B = x_obs.size(0)

  # 1. Pérdida de datos ordinal CORAL (BCE acumulativa)
  num_classes = logits.size(1) + 1
  levels = torch.arange(num_classes - 1, device=device).float()
  binary_targets = (y_people.unsqueeze(1).float() > levels).float()
  bce_loss = F.binary_cross_entropy_with_logits(
      logits, binary_targets, reduction='none'
  )
  loss_data = torch.sum(bce_loss, dim=1).mean()

  # 2. Reconstrucción Física Supervisada e Interna (Fourier Ops)
  # Construcción del CSI complejo observado: H_obs = Amp * exp(j * Phase)
  amp_obs = x_obs[:, 0, :, :]
  phase_obs = x_obs[:, 1, :, :]
  H_obs_complex = torch.complex(
      amp_obs * torch.cos(phase_obs), amp_obs * torch.sin(phase_obs)
  )

  # Reconstrucción predicha compleja
  amp_hat = H_hat[:, 0, :, :]
  phase_hat = H_hat[:, 1, :, :]
  H_hat_complex = torch.complex(
      amp_hat * torch.cos(phase_hat), amp_hat * torch.sin(phase_hat)
  )

  # A. Target CIR y Target Doppler calculados directamente sobre los datos observados
  h_target = torch.fft.ifft(
      H_obs_complex, dim=-1
  )  # Target CIR: IDFT en Subportadoras
  D_target = torch.fft.fft(
      h_target, dim=-2
  )  # Target Doppler: DFT en el Tiempo

  # B. Reconstrucción Predicha (CIR y Doppler) por Fourier Autograd
  h_hat = torch.fft.ifft(H_hat_complex, dim=-1)
  D_hat = torch.fft.fft(h_hat, dim=-2)

  # C. Pérdidas Supervisadas
  loss_rec_H = F.mse_loss(H_hat, x_obs)
  loss_rec_cir = F.mse_loss(torch.abs(h_hat), torch.abs(h_target))
  loss_rec_dop = F.mse_loss(torch.abs(D_hat), torch.abs(D_target))

  loss_physics = (
      (lambda_rec * loss_rec_H)
      + (lambda_cir * loss_rec_cir)
      + (lambda_dop * loss_rec_dop)
  )

  # 3. Pérdida de Energía Dispersada (RCS)
  latent_norms = torch.norm(latent_field, p=2, dim=1)
  target_energy = gamma * y_people.float()
  loss_energy = torch.mean((latent_norms - target_energy) ** 2)

  # 4. Regularización de Monotonicidad Suave (L_mono)
  if lambda_mono > 0.0 and latent_field.size(0) > 1:
    z_phys = (latent_norms - latent_norms.min()) / (
        latent_norms.max() - latent_norms.min() + 1e-8
    )

    sort_idx = torch.argsort(y_people)
    y_sorted = y_people[sort_idx]
    z_sorted = z_phys[sort_idx]

    diff_y = y_sorted[1:] - y_sorted[:-1]
    diff_z = z_sorted[1:] - z_sorted[:-1]
    valid_neighbors = diff_y > 0

    if torch.any(valid_neighbors):
      loss_mono_neighbors = torch.mean(
          F.relu(-(diff_z[valid_neighbors] - margin))
      )
    else:
      loss_mono_neighbors = torch.tensor(0.0, device=device)

    if mix_ratio > 0.0:
      idx_i = torch.randint(0, B, (B,), device=device)
      idx_j = torch.randint(0, B, (B,), device=device)
      pair_mask = y_people[idx_j] > y_people[idx_i]

      if torch.any(pair_mask):
        dz_random = z_phys[idx_j[pair_mask]] - z_phys[idx_i[pair_mask]]
        loss_mono_random = torch.mean(F.relu(-(dz_random - margin)))
      else:
        loss_mono_random = torch.tensor(0.0, device=device)

      loss_mono = ((1.0 - mix_ratio) * loss_mono_neighbors) + (
          mix_ratio * loss_mono_random
      )
    else:
      loss_mono = loss_mono_neighbors
  else:
    loss_mono = torch.tensor(0.0, device=device)

  loss_total = (
      loss_data
      + loss_physics
      + (lambda_energy * loss_energy)
      + (lambda_mono * loss_mono)
  )

  return loss_total, loss_data, loss_physics, loss_energy, loss_mono

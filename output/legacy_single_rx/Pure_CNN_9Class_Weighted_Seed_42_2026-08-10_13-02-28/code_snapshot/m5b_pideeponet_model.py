import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# CONFIGURACIÓN DE RUTAS Y DISPOSITIVO (PORTABLE LINUX / WINDOWS)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TENSORS_DIR = DATA_DIR / "processed_tensors"
META_PATH = TENSORS_DIR / "dataset_mc1_rx1_80mhz_meta.npz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. PYTORCH DATASET CON LECTURA ZERO-COPY DESDE DISCO (MEMMAP EN FLOAT16)
# ==============================================================================
class MemmapCSIDataset(Dataset):
    def __init__(self, dat_path, shape, starts, y_people, y_empty, window_size=25):
        self.dat_path = str(dat_path)
        self.shape = tuple(shape)
        self.starts = starts
        self.y_people = torch.tensor(y_people, dtype=torch.long)
        self.y_empty = torch.tensor(y_empty, dtype=torch.float32)
        self.window_size = window_size
        self.memmap = None

    def _init_memmap(self):
        if self.memmap is None:
            # Lectura estricta en float16 coincidente con el Módulo 5a
            self.memmap = np.memmap(self.dat_path, dtype='float16', mode='r', shape=self.shape)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        self._init_memmap()
        st = self.starts[idx]
        # Extraer ventana de 25 tramas y convertir a float32 para PyTorch
        win = self.memmap[st : st + self.window_size, :, :].astype(np.float32)
        # Reordenar ejes para Conv2D: (Canales, Tiempo, Subportadoras) -> (2, 25, 241)
        x_tensor = torch.from_numpy(win).permute(2, 0, 1).contiguous()
        return x_tensor, self.y_people[idx], self.y_empty[idx]

# ==============================================================================
# 2. ARQUITECTURA PI-DEEPONET
# ==============================================================================
class BranchNet(nn.Module):
    def __init__(self, latent_dim=128):
        super(BranchNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(3, 5), padding=(1, 2))
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 5), padding=(1, 2))
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 5), padding=(1, 2))
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.act = nn.Mish()

        self.fc = nn.Linear(64 * 3 * 30, latent_dim)

    def forward(self, x):
        x = self.pool(self.act(self.bn1(self.conv1(x))))
        x = self.pool(self.act(self.bn2(self.conv2(x))))
        x = self.pool(self.act(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

class TrunkNet(nn.Module):
    def __init__(self, latent_dim=128):
        super(TrunkNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Mish(),
            nn.Linear(64, 128),
            nn.Mish(),
            nn.Linear(128, latent_dim),
            nn.Mish()
        )

    def forward(self, grid_2d):
        return self.net(grid_2d)

class PIDeepONet(nn.Module):
    def __init__(self, num_classes=11, latent_dim=128):
        super(PIDeepONet, self).__init__()
        self.latent_dim = latent_dim
        self.branch = BranchNet(latent_dim=latent_dim)
        self.trunk = TrunkNet(latent_dim=latent_dim)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

        t_coords = np.linspace(0, 1, 25, dtype=np.float32)
        s_coords = np.linspace(0, 1, 241, dtype=np.float32)
        grid_t, grid_s = np.meshgrid(t_coords, s_coords, indexing='ij')
        grid_2d = np.stack([grid_t.ravel(), grid_s.ravel()], axis=-1)

        self.register_buffer("grid_2d", torch.from_numpy(grid_2d))

    def forward(self, x_branch):
        b_out = self.branch(x_branch)
        t_out = self.trunk(self.grid_2d)
        latent_field = torch.matmul(b_out, t_out.T)
        field_summary = b_out * torch.mean(t_out, dim=0, keepdim=True)
        logits = self.classifier(field_summary)
        return logits, latent_field

# ==============================================================================
# 3. FUNCIÓN DE LOSS TOTAL
# ==============================================================================
def compute_pi_loss(logits, latent_field, y_people, y_empty, lambda_phys=0.01, class_weights=None):
    loss_data = F.cross_entropy(logits, y_people, weight=class_weights)
    empty_mask = (y_empty == 1.0)
    if torch.any(empty_mask):
        empty_field = latent_field[empty_mask]
        loss_phys = torch.mean(empty_field ** 2)
    else:
        loss_phys = torch.tensor(0.0, device=logits.device)

    loss_total = loss_data + lambda_phys * loss_phys
    return loss_total, loss_data, loss_phys

if __name__ == "__main__":
    print(f"=== VERIFICACIÓN DE ARQUITECTURA PI-DEEPONET ===")
    print(f"Dispositivo de ejecución: {DEVICE}")

    if not META_PATH.exists():
        print(f"Aviso: {META_PATH} no se encontró para prueba local.")
    else:
        meta = np.load(META_PATH)
        num_classes = len(np.unique(meta['train_y_people']))
        print(f"Número de clases detectadas: {num_classes}")

        model = PIDeepONet(num_classes=num_classes, latent_dim=128).to(DEVICE)
        dummy_x = torch.randn(4, 2, 25, 241).to(DEVICE)
        dummy_y_people = torch.tensor([0, 1, 2, 0]).to(DEVICE)
        dummy_y_empty = torch.tensor([1.0, 0.0, 0.0, 1.0]).to(DEVICE)

        logits, latent_field = model(dummy_x)
        loss_total, loss_data, loss_phys = compute_pi_loss(logits, latent_field, dummy_y_people, dummy_y_empty)

        print(f"Entrada Branch: {dummy_x.shape} | Logits: {logits.shape} | Latent Field: {latent_field.shape}")
        print(f"Loss Total: {loss_total.item():.4f} (Data: {loss_data.item():.4f}, Phys: {loss_phys.item():.4f})")
        print("¡Prueba de arquitectura exitosa!")

# 🔬 Reporte Científico de Experimento: 2026-08-12_09-51-16

## 1. Resumen de Ejecución
- **Duración Total:** 46.57 minutos (2794.2 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-06
- **Test Loss:** 5.885331347262733
- **Test Accuracy_Percent:** 11.089535402164987

---
## 2. Entorno de Hardware y Software
- **Sistema Operativo:** Linux 6.8.0-136-generic (x86_64)
- **Python Version:** 3.10.12
- **PyTorch Version:** 2.5.1+cu121
- **GPU:** NVIDIA RTX 4000 Ada Generation (19.54 GB VRAM)
- **CUDA Version:** 12.1

### Bibliotecas Clave Instaladas
| Biblioteca | Versión |
| :--- | :--- |
| `torch` | `2.5.1+cu121` |
| `numpy` | `2.2.6` |
| `pandas` | `2.3.3` |
| `scipy` | `1.15.3` |
| `scikit-learn` | `1.7.2` |
| `torchinfo` | `1.8.0` |
| `torchviz` | `0.0.3` |
| `onnx` | `1.22.0` |

*(Ver lista completa de 75 bibliotecas en `environment_libraries.csv`)*

---
## 3. Configuración e Hiperparámetros
| Parámetro | Valor |
| :--- | :--- |
| `SEED` | `42` |
| `NUM_CLASSES` | `9` |
| `BATCH_SIZE` | `256` |
| `NUM_WORKERS` | `4` |
| `EPOCHS` | `15` |
| `LEARNING_RATE` | `0.001` |
| `LAMBDA_PHYS` | `0.0` |
| `LAMBDA_TEMP` | `0.01` |
| `LAMBDA_ENERGY` | `0.01` |
| `GAMMA` | `1.0` |
| `MODEL` | `PI-DeepONet_CORAL_Ordinal_Energy` |
| `DEVICE` | `cuda` |

---
## 4. Trazabilidad de Archivos de Datos (Hashes SHA-256)
| Identificador | Ruta | SHA-256 Hash |
| :--- | :--- | :--- |
| `metadata` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/dataset_mc1_rx1_80mhz_meta.npz` | `39ee2e7ece7c366e...` |
| `X_train` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_train_frames.dat` | `4b6d73f1d7ab3d55...` |
| `X_val` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_val_frames.dat` | `3a485662986e2f80...` |
| `X_test` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_test_frames.dat` | `f2cbc80908d965ed...` |

---
## 5. Historial de Entrenamiento por Época
|   epoch |   train_loss |   train_acc |   val_loss |   val_acc |   tr_loss_temp |   val_loss_temp |   tr_loss_energy |   val_loss_energy |
|--------:|-------------:|------------:|-----------:|----------:|---------------:|----------------:|-----------------:|------------------:|
|       1 |      9.18705 |     30.4214 |    2.96106 |   24.2457 |        1.9e-05 |        0.000129 |       623.591    |          49.7678  |
|       2 |      4.71588 |     40.5034 |    2.4592  |   36.9117 |        1e-05   |        7e-06    |       258.959    |           2.53291 |
|       3 |      1.85814 |     46.0197 |    1.77838 |   43.7411 |        1e-05   |        8e-06    |         2.74088  |           2.30997 |
|       4 |      2.37331 |     46.5734 |    1.72751 |   46.8258 |        1e-05   |        6e-06    |        59.4727   |           3.19285 |
|       5 |      1.93216 |     51.1645 |    1.61651 |   49.2455 |        7e-06   |        4e-06    |        41.7803   |           1.75434 |
|       6 |      1.38051 |     54.222  |    1.54936 |   48.2311 |        7e-06   |        8e-06    |         2.08696  |           2.61469 |
|       7 |      1.38126 |     55.7103 |    1.49773 |   48.389  |        1e-05   |        9e-06    |         8.07183  |           2.34234 |
|       8 |      1.24121 |     57.5293 |    1.52844 |   49.0093 |        1.1e-05 |        9e-06    |         2.01149  |           2.91534 |
|       9 |      1.19663 |     58.7389 |    1.57367 |   46.7176 |        1.3e-05 |        1.4e-05  |         2.07424  |           2.05889 |
|      10 |      1.16571 |     59.5669 |    1.5856  |   47.7652 |        1.6e-05 |        1.4e-05  |         1.99939  |           1.66135 |
|      11 |      1.11193 |     60.917  |    1.59778 |   46.8529 |        1.6e-05 |        1.5e-05  |         1.30368  |           1.52371 |
|      12 |      1.09785 |     61.288  |    1.54994 |   49.4502 |        1.5e-05 |        1.3e-05  |         1.32102  |           1.88114 |
|      13 |      1.08747 |     61.6473 |    1.61338 |   47.8439 |        1.5e-05 |        1.5e-05  |         1.30712  |           1.36737 |
|      14 |      1.06004 |     62.3473 |    1.6177  |   46.6768 |        1.6e-05 |        1.5e-05  |         0.989234 |           1.1136  |
|      15 |      1.05321 |     62.5493 |    1.6262  |   46.6    |        1.6e-05 |        1.5e-05  |         1.00104  |           1.25578 |

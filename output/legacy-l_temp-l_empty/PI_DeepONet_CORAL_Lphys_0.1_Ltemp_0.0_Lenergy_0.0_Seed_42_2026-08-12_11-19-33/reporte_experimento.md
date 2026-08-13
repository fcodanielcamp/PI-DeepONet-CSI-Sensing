# 🔬 Reporte Científico de Experimento: 2026-08-12_11-19-33

## 1. Resumen de Ejecución
- **Duración Total:** 46.69 minutos (2801.1 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-06
- **Test Loss:** 6.76766608812266
- **Test Accuracy_Percent:** 10.518198717964337

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
| `LAMBDA_PHYS` | `0.1` |
| `LAMBDA_TEMP` | `0.0` |
| `LAMBDA_ENERGY` | `0.0` |
| `GAMMA` | `0.0` |
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
|       1 |     1.96498  |     48.3895 |    1.52279 |   50.6624 |        3.2e-05 |         8e-06   |          3171.3  |           4609.61 |
|       2 |     1.24924  |     56.9899 |    1.39002 |   51.2434 |        6e-06   |         1.3e-05 |          2956.45 |           2666.45 |
|       3 |     1.15846  |     59.343  |    1.42708 |   50.3114 |        7e-06   |         0       |          2172.12 |           1851.74 |
|       4 |     1.11099  |     60.7558 |    1.42518 |   51.2113 |        7e-06   |         4e-06   |          2031.81 |           2179.66 |
|       5 |     1.07846  |     61.7926 |    1.51581 |   46.2786 |        1e-05   |         1e-06   |          1857.93 |           1998.21 |
|       6 |     1.02744  |     63.1908 |    1.52914 |   47.1425 |        5e-06   |         9e-06   |          1582.07 |           1451.33 |
|       7 |     1.01407  |     63.5947 |    1.49894 |   49.2194 |        3e-06   |         1e-05   |          1553.43 |           1512.37 |
|       8 |     1.00339  |     63.9339 |    1.60456 |   45.7342 |        5e-06   |         0       |          1503.99 |           1448.24 |
|       9 |     0.978508 |     64.6048 |    1.61208 |   46.867  |        0       |         0       |          1454.27 |           1402.42 |
|      10 |     0.97259  |     64.7853 |    1.59069 |   46.7045 |        0       |         0       |          1489.19 |           1747.19 |
|      11 |     0.968411 |     64.9619 |    1.65171 |   45.8731 |        1e-06   |         1e-06   |          1460.31 |           1437.37 |
|      12 |     0.954141 |     65.3283 |    1.57959 |   48.3267 |        0       |         0       |          1464.68 |           1585.88 |
|      13 |     0.951799 |     65.4159 |    1.57881 |   47.8782 |        0       |         0       |          1504.23 |           1680.4  |
|      14 |     0.949494 |     65.5172 |    1.62327 |   47.0361 |        0       |         0       |          1503.05 |           1599.53 |
|      15 |     0.941675 |     65.7194 |    1.64134 |   45.71   |        0       |         0       |          1505.98 |           1693.73 |

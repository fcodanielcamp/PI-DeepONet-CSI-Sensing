# 🔬 Reporte Científico de Experimento: 2026-08-12_00-20-59

## 1. Resumen de Ejecución
- **Duración Total:** 46.10 minutos (2765.8 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-06
- **Test Loss:** 6.741673600786047
- **Test Accuracy_Percent:** 10.52124422267756

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
| `LAMBDA_TEMP` | `0.01` |
| `MODEL` | `PI-DeepONet_CORAL_Ordinal` |
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
|   epoch |   train_loss |   train_acc |   val_loss |   val_acc |   tr_loss_temp |   val_loss_temp |
|--------:|-------------:|------------:|-----------:|----------:|---------------:|----------------:|
|       1 |     1.96416  |     48.4073 |    1.51882 |   50.7831 |        3.1e-05 |           1e-05 |
|       2 |     1.24798  |     57.0044 |    1.40531 |   50.134  |        6e-06   |           7e-06 |
|       3 |     1.15862  |     59.3282 |    1.41893 |   50.1458 |        5e-06   |           1e-06 |
|       4 |     1.11231  |     60.7238 |    1.4236  |   51.2067 |        6e-06   |           1e-06 |
|       5 |     1.0804   |     61.6949 |    1.47017 |   47.4831 |        1.6e-05 |           4e-06 |
|       6 |     1.02915  |     63.1374 |    1.54166 |   46.9057 |        2e-06   |           1e-06 |
|       7 |     1.01573  |     63.5546 |    1.49399 |   48.9193 |        2e-06   |           0     |
|       8 |     1.00561  |     63.864  |    1.52711 |   46.8827 |        4e-06   |           3e-06 |
|       9 |     0.980768 |     64.5498 |    1.5857  |   46.9121 |        0       |           1e-06 |
|      10 |     0.974902 |     64.7115 |    1.59125 |   46.8192 |        0       |           0     |
|      11 |     0.970643 |     64.8764 |    1.61907 |   46.4448 |        1e-06   |           0     |
|      12 |     0.956448 |     65.2482 |    1.58049 |   48.233  |        0       |           0     |
|      13 |     0.954174 |     65.3532 |    1.57468 |   47.9735 |        0       |           0     |
|      14 |     0.951809 |     65.4344 |    1.60337 |   47.0787 |        0       |           0     |
|      15 |     0.94371  |     65.6565 |    1.63449 |   45.8305 |        0       |           0     |

# 🔬 Reporte Científico de Experimento: 2026-08-12_06-27-30

## 1. Resumen de Ejecución
- **Duración Total:** 46.42 minutos (2785.1 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-06
- **Test Loss:** 7.786285168013819
- **Test Accuracy_Percent:** 10.333884772720074

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
|       1 |     1.80287  |     50.172  |    1.46818 |   48.0326 |       0.001172 |        0.000203 |      3.43449e+07 |       3.11013e+07 |
|       2 |     1.18495  |     58.4942 |    1.45479 |   48.2315 |       0.00031  |        0.00026  |      1.61757e+07 |       1.13261e+07 |
|       3 |     1.10321  |     60.8235 |    1.35506 |   50.6456 |       0.000406 |        0.000538 |      7.21192e+06 |       6.62853e+06 |
|       4 |     1.05867  |     62.123  |    1.43866 |   48.5205 |       0.000191 |        0.000278 |      4.79239e+06 |       4.16148e+06 |
|       5 |     1.02824  |     63.1233 |    1.44182 |   48.732  |       0.000114 |        9e-05    |      3.3957e+06  |       3.75727e+06 |
|       6 |     1.00605  |     63.7529 |    1.58384 |   45.3954 |       8.8e-05  |        4.5e-05  |      2.8091e+06  |       3.61636e+06 |
|       7 |     0.95549  |     65.3481 |    1.50339 |   48.0947 |       2.3e-05  |        1.8e-05  |      2.86243e+06 |       3.44507e+06 |
|       8 |     0.943455 |     65.6799 |    1.56909 |   46.6457 |       1.9e-05  |        1.7e-05  |      2.72129e+06 |       3.33637e+06 |
|       9 |     0.934557 |     65.9885 |    1.67596 |   45.4536 |       1.9e-05  |        2.8e-05  |      2.59401e+06 |       3.31965e+06 |
|      10 |     0.907302 |     66.7951 |    1.61627 |   46.2928 |       8e-06    |        9e-06    |      2.76085e+06 |       3.42039e+06 |
|      11 |     0.901419 |     66.9873 |    1.68385 |   45.2104 |       8e-06    |        7e-06    |      2.75476e+06 |       3.31817e+06 |
|      12 |     0.896249 |     67.1235 |    1.59401 |   48.1254 |       6e-06    |        5e-06    |      2.74773e+06 |       3.2864e+06  |
|      13 |     0.881316 |     67.6153 |    1.61346 |   46.7668 |       3e-06    |        4e-06    |      2.7977e+06  |       3.55002e+06 |
|      14 |     0.877989 |     67.6874 |    1.6889  |   46.2506 |       3e-06    |        2e-06    |      2.85028e+06 |       3.60092e+06 |
|      15 |     0.875174 |     67.7715 |    1.69701 |   45.1949 |       3e-06    |        2e-06    |      2.89466e+06 |       3.61682e+06 |

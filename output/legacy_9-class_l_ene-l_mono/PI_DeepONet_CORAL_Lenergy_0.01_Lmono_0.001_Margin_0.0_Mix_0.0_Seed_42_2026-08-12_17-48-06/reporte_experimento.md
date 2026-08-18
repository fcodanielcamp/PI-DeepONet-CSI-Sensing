# 🔬 Reporte Científico de Experimento: 2026-08-12_17-48-06

## 1. Resumen de Ejecución
- **Duración Total:** 46.15 minutos (2769.1 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-06
- **Test Loss:** 6.379758140522252
- **Test Accuracy_Percent:** 10.920448980486842

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
| `EPOCHS` | `15` |
| `LEARNING_RATE` | `0.001` |
| `LAMBDA_ENERGY` | `0.01` |
| `LAMBDA_MONO` | `0.001` |
| `MARGIN` | `0.0` |
| `WARMUP_EPOCHS` | `5` |
| `MIX_RATIO` | `0.0` |
| `GAMMA` | `1.0` |
| `MODEL` | `PI-DeepONet_CORAL_Monotonic_Energy` |
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
|   epoch |   train_loss |   train_acc |   val_loss |   val_acc |   tr_loss_energy |   val_loss_energy |   tr_loss_mono |   val_loss_mono |
|--------:|-------------:|------------:|-----------:|----------:|-----------------:|------------------:|---------------:|----------------:|
|       1 |      9.05903 |     30.9086 |    2.09059 |   43.738  |       613.079    |           4.52841 |       0        |        0        |
|       2 |      7.63655 |     35.9995 |    2.33391 |   35.2348 |       523.63     |           2.43527 |       0        |        0        |
|       3 |      1.8926  |     45.9548 |    1.71746 |   46.7894 |         5.82742  |           4.28604 |       0        |        0        |
|       4 |      2.06802 |     46.8391 |    1.64389 |   47.6945 |        30.9628   |           1.97866 |       0        |        0        |
|       5 |      1.46507 |     53.0418 |    1.55116 |   48.3278 |         4.97915  |           3.06208 |       0        |        0        |
|       6 |      1.30038 |     56.0996 |    1.43223 |   50.7831 |         2.10673  |           2.09643 |       0.050515 |        0.001915 |
|       7 |      1.22264 |     58.12   |    1.47165 |   49.3593 |         2.21672  |           2.92143 |       0.050186 |        0.003033 |
|       8 |      1.17653 |     59.3843 |    1.45867 |   50.3054 |         2.11363  |           2.47405 |       0.048384 |        0.002856 |
|       9 |      1.14588 |     60.2891 |    1.58677 |   45.0757 |         2.02793  |           2.77754 |       0.046766 |        0.002072 |
|      10 |      1.09434 |     61.5172 |    1.54206 |   47.521  |         1.31586  |           1.17042 |       0.036923 |        0.000636 |
|      11 |      1.08021 |     61.9463 |    1.63324 |   45.4662 |         1.32035  |           1.40438 |       0.037453 |        0.000665 |
|      12 |      1.06966 |     62.2518 |    1.52623 |   48.3702 |         1.31526  |           2.08083 |       0.03643  |        0.002551 |
|      13 |      1.04387 |     62.8841 |    1.52082 |   49.0542 |         0.981455 |           1.2016  |       0.031797 |        0.000635 |
|      14 |      1.03752 |     63.0858 |    1.61489 |   45.8696 |         0.990917 |           1.05133 |       0.03191  |        0.000625 |
|      15 |      1.03249 |     63.2338 |    1.62178 |   46.3282 |         0.993258 |           1.28461 |       0.031468 |        0.000497 |

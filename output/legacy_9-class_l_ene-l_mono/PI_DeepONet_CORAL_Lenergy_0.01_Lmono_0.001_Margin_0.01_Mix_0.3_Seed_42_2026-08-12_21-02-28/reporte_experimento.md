# 🔬 Reporte Científico de Experimento: 2026-08-12_21-02-28

## 1. Resumen de Ejecución
- **Duración Total:** 46.12 minutos (2767.3 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-06
- **Test Loss:** 6.218527697034662
- **Test Accuracy_Percent:** 11.15154187812621

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
| `MARGIN` | `0.01` |
| `WARMUP_EPOCHS` | `8` |
| `MIX_RATIO` | `0.3` |
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
|       1 |      9.16416 |     30.251  |    2.20218 |   42.2412 |        620.104   |           2.17892 |       0        |        0        |
|       2 |      5.6103  |     38.6617 |    2.52062 |   31.4427 |        337.247   |           3.78579 |       0        |        0        |
|       3 |      1.88288 |     46.1035 |    1.76838 |   43.7746 |          4.70182 |           4.39211 |       0        |        0        |
|       4 |      2.1361  |     47.9615 |    1.63395 |   47.3863 |         42.9856  |           3.19242 |       0        |        0        |
|       5 |      1.40958 |     53.8256 |    1.5968  |   47.5034 |          2.63467 |           4.23754 |       0        |        0        |
|       6 |      1.30226 |     56.357  |    1.44064 |   51.5544 |          2.49544 |           2.16008 |       0        |        0        |
|       7 |      1.23612 |     57.9899 |    1.47227 |   50.0647 |          2.20879 |           2.66448 |       0        |        0        |
|       8 |      1.19489 |     59.019  |    1.56103 |   46.8068 |          2.10289 |           1.88545 |       0        |        0        |
|       9 |      1.164   |     59.787  |    1.61421 |   45.0281 |          2.02602 |           2.13673 |       0.042769 |        0.002433 |
|      10 |      1.11048 |     61.0489 |    1.62816 |   45.9706 |          1.35199 |           1.37334 |       0.03318  |        0.000993 |
|      11 |      1.09535 |     61.5056 |    1.6327  |   45.0502 |          1.36561 |           1.21366 |       0.03375  |        0.001198 |
|      12 |      1.08358 |     61.8328 |    1.52353 |   50.0579 |          1.3663  |           1.6872  |       0.032865 |        0.001389 |
|      13 |      1.05583 |     62.543  |    1.52829 |   47.9554 |          1.03045 |           1.20182 |       0.028573 |        0.000975 |
|      14 |      1.04905 |     62.7289 |    1.59461 |   46.1997 |          1.0435  |           1.11873 |       0.028488 |        0.000826 |
|      15 |      1.04385 |     62.8712 |    1.57849 |   46.5567 |          1.04662 |           1.15038 |       0.028409 |        0.000863 |

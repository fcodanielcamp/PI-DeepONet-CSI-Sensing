# 🔬 Reporte Científico de Experimento: 2026-08-18_11-31-53

## 1. Resumen de Ejecución
- **Duración Total:** 42.22 minutos (2533.1 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-01B
- **Test Loss:** 9.522411156915783
- **Test Accuracy_Percent:** 15.85341791955952

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
| `NUM_CLASSES` | `5` |
| `BATCH_SIZE` | `256` |
| `EPOCHS` | `15` |
| `LEARNING_RATE` | `0.001` |
| `LAMBDA_ENERGY` | `0.0` |
| `LAMBDA_MONO` | `0.0` |
| `MARGIN` | `0.0` |
| `WARMUP_EPOCHS` | `0` |
| `MIX_RATIO` | `0.0` |
| `GAMMA` | `1.0` |
| `MODEL` | `PI-DeepONet_CORAL_Monotonic_Energy` |
| `DEVICE` | `cuda` |

---
## 4. Trazabilidad de Archivos de Datos (Hashes SHA-256)
| Identificador | Ruta | SHA-256 Hash |
| :--- | :--- | :--- |
| `metadata` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/dataset_mc1_rx1_80mhz_meta.npz` | `b4779fcd95fefadd...` |
| `X_train` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_train_frames.dat` | `17b8567418dad173...` |
| `X_val` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_val_frames.dat` | `a323d19d9f0be89a...` |
| `X_test` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_test_frames.dat` | `5a03477aa2b0d325...` |

---
## 5. Historial de Entrenamiento por Época
|   epoch |   train_loss |   train_acc |   val_loss |   val_acc |   tr_loss_energy |   val_loss_energy |   tr_loss_mono |   val_loss_mono |
|--------:|-------------:|------------:|-----------:|----------:|-----------------:|------------------:|---------------:|----------------:|
|       1 |     1.05203  |     59.5249 |    1.42295 |   51.3948 |      6.32363e+07 |       6.60021e+07 |              0 |               0 |
|       2 |     0.832703 |     66.2437 |    1.37242 |   55.8273 |      4.15818e+07 |       2.9198e+07  |              0 |               0 |
|       3 |     0.772277 |     68.4792 |    1.44516 |   52.2565 |      2.07429e+07 |       1.28869e+07 |              0 |               0 |
|       4 |     0.734855 |     69.8891 |    1.38163 |   53.8798 |      1.23143e+07 |       9.78863e+06 |              0 |               0 |
|       5 |     0.706726 |     70.96   |    1.47492 |   52.0047 |      8.19723e+06 |       7.36331e+06 |              0 |               0 |
|       6 |     0.660698 |     72.7496 |    1.49426 |   52.4985 |      7.74045e+06 |       7.59593e+06 |              0 |               0 |
|       7 |     0.647422 |     73.2292 |    1.53925 |   53.4201 |      7.30863e+06 |       7.00672e+06 |              0 |               0 |
|       8 |     0.637324 |     73.6515 |    1.57717 |   51.9826 |      6.99932e+06 |       6.25123e+06 |              0 |               0 |
|       9 |     0.613162 |     74.5739 |    1.52184 |   54.1659 |      7.11171e+06 |       6.47785e+06 |              0 |               0 |
|      10 |     0.606826 |     74.8052 |    1.65335 |   50.9857 |      7.17025e+06 |       6.85046e+06 |              0 |               0 |
|      11 |     0.600956 |     75.0702 |    1.58748 |   52.9321 |      7.1752e+06  |       7.02074e+06 |              0 |               0 |
|      12 |     0.587847 |     75.5508 |    1.66307 |   52.7453 |      7.39606e+06 |       7.34173e+06 |              0 |               0 |
|      13 |     0.584738 |     75.7025 |    1.64474 |   54.1357 |      7.55166e+06 |       7.03343e+06 |              0 |               0 |
|      14 |     0.581715 |     75.8298 |    1.67115 |   53.0862 |      7.73392e+06 |       7.46908e+06 |              0 |               0 |
|      15 |     0.574177 |     76.1481 |    1.68025 |   52.4622 |      8.04894e+06 |       7.82126e+06 |              0 |               0 |

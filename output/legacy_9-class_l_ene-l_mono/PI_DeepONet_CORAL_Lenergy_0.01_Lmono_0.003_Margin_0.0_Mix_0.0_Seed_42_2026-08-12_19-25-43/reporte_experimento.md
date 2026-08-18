# 🔬 Reporte Científico de Experimento: 2026-08-12_19-25-43

## 1. Resumen de Ejecución
- **Duración Total:** 45.98 minutos (2759.0 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-06
- **Test Loss:** 6.314356560548987
- **Test Accuracy_Percent:** 11.074673339164459

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
| `LAMBDA_MONO` | `0.003` |
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
|       1 |      9.08581 |     30.8299 |    2.07401 |   45.1642 |       615.388    |           2.72749 |       0        |        0        |
|       2 |      5.51959 |     38.8563 |    2.51085 |   34.1095 |       330.786    |           4.02916 |       0        |        0        |
|       3 |      1.91309 |     45.4275 |    1.79924 |   43.8889 |         3.66147  |           3.64672 |       0        |        0        |
|       4 |      2.24562 |     46.7774 |    1.70169 |   46.445  |        46.8366   |           2.18684 |       0        |        0        |
|       5 |      1.50245 |     52.7372 |    1.57863 |   48.3255 |         6.52546  |           4.37904 |       0        |        0        |
|       6 |      1.31856 |     55.7812 |    1.4765  |   49.2563 |         2.07001  |           2.22821 |       0.050709 |        0.003242 |
|       7 |      1.24131 |     57.7058 |    1.46685 |   49.0348 |         2.15586  |           1.99821 |       0.050664 |        0.002246 |
|       8 |      1.1952  |     58.8827 |    1.51593 |   47.8677 |         2.10081  |           1.55009 |       0.048785 |        0.001493 |
|       9 |      1.16321 |     59.7151 |    1.63366 |   45.234  |         1.98792  |           1.71239 |       0.046297 |        0.001924 |
|      10 |      1.13943 |     60.3816 |    1.57318 |   46.7964 |         1.90856  |           1.81547 |       0.044887 |        0.001689 |
|      11 |      1.09125 |     61.5094 |    1.6602  |   44.6021 |         1.23696  |           1.33239 |       0.036828 |        0.001383 |
|      12 |      1.07963 |     61.8301 |    1.5031  |   48.2427 |         1.26005  |           1.50099 |       0.036184 |        0.001688 |
|      13 |      1.07105 |     62.1076 |    1.58139 |   47.431  |         1.25269  |           1.48708 |       0.036088 |        0.001302 |
|      14 |      1.0459  |     62.7256 |    1.62319 |   45.6397 |         0.964295 |           1.01646 |       0.032207 |        0.000965 |
|      15 |      1.04066 |     62.8293 |    1.60736 |   45.8052 |         0.969817 |           1.24982 |       0.031916 |        0.001236 |

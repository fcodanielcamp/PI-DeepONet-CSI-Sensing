# 🔬 Reporte Científico de Experimento: 2026-08-20_10-21-47

## 1. Resumen de Ejecución
- **Duración Total:** 87.87 minutos (5272.3 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-01B
- **Test Loss:** 8.693101867946348
- **Test Accuracy_Percent:** 15.823705963050397

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
| `LAMBDA_MONO` | `0.001` |
| `LAMBDA_REC` | `1.0` |
| `LAMBDA_CIR` | `0.01` |
| `LAMBDA_DOP` | `0.01` |
| `MODEL` | `PI-DeepONet_Physics_Fourier` |
| `DEVICE` | `cuda` |

---
## 4. Trazabilidad de Archivos de Datos (Hashes SHA-256)
| Identificador | Ruta | SHA-256 Hash |
| :--- | :--- | :--- |
| `metadata` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/dataset_mc1_rx1_80mhz_meta.npz` | `532cc6f8e5410fa9...` |
| `X_train` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_train_frames.dat` | `e29dc479c9c23cbb...` |
| `X_val` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_val_frames.dat` | `0513eb4ed4a856a7...` |
| `X_test` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_test_frames.dat` | `e2b1497acf70184b...` |

---
## 5. Historial de Entrenamiento por Época
|   epoch |   train_loss |   train_acc |   val_loss |   val_acc |   tr_loss_phys |   val_loss_phys |   tr_loss_energy |   val_loss_energy |
|--------:|-------------:|------------:|-----------:|----------:|---------------:|----------------:|-----------------:|------------------:|
|       1 |     1.29777  |     60.0992 |    1.43608 |   55.4833 |       0.269592 |        0.263384 |          4.95038 |           7.40743 |
|       2 |     1.02909  |     68.5505 |    1.74172 |   50.1337 |       0.259257 |        0.261577 |          7.04221 |           8.09916 |
|       3 |     0.962166 |     70.9872 |    1.70111 |   54.6524 |       0.257864 |        0.261226 |          7.50571 |           8.45574 |
|       4 |     0.923327 |     72.4577 |    1.94559 |   49.6543 |       0.257321 |        0.260095 |          7.62624 |           9.00727 |
|       5 |     0.873647 |     74.2499 |    1.81989 |   53.9321 |       0.256028 |        0.258916 |          7.84384 |           8.84315 |
|       6 |     0.857751 |     74.9195 |    1.91362 |   53.5849 |       0.255882 |        0.258814 |          7.92056 |           8.59873 |
|       7 |     0.844901 |     75.4122 |    1.94688 |   53.8122 |       0.255811 |        0.258778 |          7.93971 |           8.8694  |
|       8 |     0.819413 |     76.3664 |    1.99586 |   53.2295 |       0.255348 |        0.258344 |          7.96455 |           8.70937 |
|       9 |     0.811506 |     76.7044 |    2.05154 |   50.9697 |       0.255242 |        0.258374 |          8.03112 |           9.01561 |
|      10 |     0.804823 |     76.9552 |    2.05162 |   52.4565 |       0.255138 |        0.258119 |          8.12792 |           8.9679  |
|      11 |     0.790994 |     77.5047 |    2.04887 |   52.8238 |       0.255299 |        0.258305 |          8.18182 |           8.77023 |
|      12 |     0.786979 |     77.65   |    2.06311 |   52.0786 |       0.255266 |        0.258296 |          8.20309 |           8.76525 |
|      13 |     0.782803 |     77.8448 |    2.13387 |   51.1229 |       0.25524  |        0.258327 |          8.21987 |           8.95251 |
|      14 |     0.774807 |     78.1422 |    2.13244 |   51.2039 |       0.255126 |        0.258157 |          8.23096 |           9.05249 |
|      15 |     0.772818 |     78.2227 |    2.1091  |   52.2401 |       0.255107 |        0.258168 |          8.23987 |           9.07805 |

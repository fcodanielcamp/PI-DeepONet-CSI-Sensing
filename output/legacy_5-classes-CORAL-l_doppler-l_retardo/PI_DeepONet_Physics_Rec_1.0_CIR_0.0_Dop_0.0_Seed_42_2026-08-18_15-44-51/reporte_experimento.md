# 🔬 Reporte Científico de Experimento: 2026-08-18_15-44-51

## 1. Resumen de Ejecución
- **Duración Total:** 50.78 minutos (3047.0 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-01B
- **Test Loss:** 10.27946993844498
- **Test Accuracy_Percent:** 15.889861491215242

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
| `LAMBDA_REC` | `1.0` |
| `LAMBDA_CIR` | `0.0` |
| `LAMBDA_DOP` | `0.0` |
| `MODEL` | `PI-DeepONet_Physics_Fourier` |
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
|   epoch |   train_loss |   train_acc |   val_loss |   val_acc |   tr_loss_phys |   val_loss_phys |   tr_loss_energy |   val_loss_energy |
|--------:|-------------:|------------:|-----------:|----------:|---------------:|----------------:|-----------------:|------------------:|
|       1 |      1.56012 |     59.8042 |    1.89041 |   50.9861 |       0.517046 |        0.518029 |      2.49439e+07 |       1.72765e+07 |
|       2 |      1.31461 |     67.298  |    1.90868 |   52.4349 |       0.510364 |        0.515988 |      9.17521e+06 |       6.45057e+06 |
|       3 |      1.2485  |     69.6978 |    1.88933 |   53.7102 |       0.509348 |        0.51559  |      4.1994e+06  |       3.36927e+06 |
|       4 |      1.21109 |     71.0752 |    1.93731 |   54.0017 |       0.508751 |        0.514699 |      2.93971e+06 |       2.61794e+06 |
|       5 |      1.18487 |     72.0885 |    1.94628 |   54.2995 |       0.50839  |        0.51521  |      2.32438e+06 |       2.41159e+06 |
|       6 |      1.16692 |     72.7572 |    1.81593 |   54.9932 |       0.508323 |        0.515019 |      1.92535e+06 |       2.0107e+06  |
|       7 |      1.15259 |     73.355  |    1.85321 |   55.3489 |       0.508271 |        0.514688 |      1.64753e+06 |       1.7228e+06  |
|       8 |      1.13871 |     73.8568 |    2.06753 |   53.37   |       0.508322 |        0.514701 |      1.46365e+06 |       1.37423e+06 |
|       9 |      1.12868 |     74.2259 |    2.02355 |   55.0401 |       0.508194 |        0.514202 |      1.33312e+06 |       1.31896e+06 |
|      10 |      1.09421 |     75.5439 |    1.9761  |   54.6532 |       0.506952 |        0.51318  |      1.36473e+06 |       1.47767e+06 |
|      11 |      1.08654 |     75.8165 |    2.01948 |   55.2143 |       0.506758 |        0.512852 |      1.38651e+06 |       1.56324e+06 |
|      12 |      1.08112 |     76.0498 |    2.10178 |   54.7364 |       0.50681  |        0.512661 |      1.35984e+06 |       1.40449e+06 |
|      13 |      1.06172 |     76.8017 |    2.17207 |   53.0283 |       0.506134 |        0.512088 |      1.42746e+06 |       1.53685e+06 |
|      14 |      1.05718 |     76.9545 |    2.14318 |   53.7401 |       0.506023 |        0.511998 |      1.4378e+06  |       1.61759e+06 |
|      15 |      1.0546  |     77.0876 |    2.19857 |   53.9936 |       0.505957 |        0.511948 |      1.44508e+06 |       1.65503e+06 |

# 🔬 Reporte Científico de Experimento: 2026-08-11_12-12-20

## 1. Resumen de Ejecución
- **Duración Total:** 13.96 minutos (837.8 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-06
- **Test Loss:** 19.504944025299928
- **Test Accuracy_Percent:** 10.002691882052837

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
| `LAMBDA_PHYS` | `0.01` |
| `LAMBDA_TEMP` | `0.001` |
| `MODEL` | `PI-DeepONet_9Class_TempLoss` |
| `DEVICE` | `cuda` |

---
## 4. Trazabilidad de Archivos de Datos (Hashes SHA-256)
| Identificador | Ruta | SHA-256 Hash |
| :--- | :--- | :--- |
| `metadata` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/dataset_mc1_rx1_80mhz_meta.npz` | `6faecd85b0b43ae7...` |
| `X_train` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_train_frames.dat` | `5da82590d4422945...` |
| `X_val` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_val_frames.dat` | `686b388a521ae147...` |
| `X_test` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_test_frames.dat` | `74471c351dde52bf...` |

---
## 5. Historial de Entrenamiento por Época
|   epoch |   train_loss |   train_acc |   val_loss |   val_acc |   tr_loss_temp |   val_loss_temp |
|--------:|-------------:|------------:|-----------:|----------:|---------------:|----------------:|
|       1 |     1.38348  |     49.9123 |    2.12867 |   32.1994 |       0.000235 |        7e-05    |
|       2 |     1.09818  |     63.0555 |    2.57981 |   24.4546 |       4.7e-05  |        4.6e-05  |
|       3 |     1.0262   |     66.0264 |    2.57114 |   28.0529 |       3.9e-05  |        0.000107 |
|       4 |     0.980626 |     67.8644 |    3.3996  |   22.1552 |       3.7e-05  |        2.4e-05  |
|       5 |     0.92124  |     70.0049 |    3.58956 |   25.1439 |       1.3e-05  |        9e-06    |
|       6 |     0.900477 |     70.7013 |    3.41998 |   25.8482 |       1.5e-05  |        1.6e-05  |
|       7 |     0.881592 |     71.2906 |    3.99086 |   22.7803 |       1.6e-05  |        2.2e-05  |
|       8 |     0.844974 |     72.5348 |    5.22234 |   21.6979 |       1.6e-05  |        1.8e-05  |
|       9 |     0.830652 |     72.9752 |    4.68386 |   22.6543 |       1.1e-05  |        9e-06    |
|      10 |     0.817867 |     73.3604 |    4.21791 |   24.8614 |       1e-05    |        8e-06    |
|      11 |     0.793593 |     74.0463 |    4.92495 |   22.8838 |       7e-06    |        9e-06    |
|      12 |     0.783603 |     74.3537 |    5.04069 |   23.366  |       7e-06    |        7e-06    |
|      13 |     0.775871 |     74.5767 |    5.04091 |   23.7541 |       7e-06    |        9e-06    |
|      14 |     0.759801 |     75.0732 |    5.21995 |   23.7715 |       7e-06    |        7e-06    |
|      15 |     0.754697 |     75.1903 |    5.29497 |   23.3274 |       6e-06    |        7e-06    |

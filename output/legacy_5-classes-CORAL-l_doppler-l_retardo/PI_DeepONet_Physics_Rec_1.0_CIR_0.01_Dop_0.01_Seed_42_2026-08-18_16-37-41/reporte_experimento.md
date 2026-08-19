# 🔬 Reporte Científico de Experimento: 2026-08-18_16-37-41

## 1. Resumen de Ejecución
- **Duración Total:** 51.70 minutos (3102.0 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-01B
- **Test Loss:** 8.353924415067949
- **Test Accuracy_Percent:** 15.833223074119726

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
| `LAMBDA_ENERGY` | `0.01` |
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
| `metadata` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/dataset_mc1_rx1_80mhz_meta.npz` | `b4779fcd95fefadd...` |
| `X_train` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_train_frames.dat` | `17b8567418dad173...` |
| `X_val` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_val_frames.dat` | `a323d19d9f0be89a...` |
| `X_test` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_test_frames.dat` | `5a03477aa2b0d325...` |

---
## 5. Historial de Entrenamiento por Época
|   epoch |   train_loss |   train_acc |   val_loss |   val_acc |   tr_loss_phys |   val_loss_phys |   tr_loss_energy |   val_loss_energy |
|--------:|-------------:|------------:|-----------:|----------:|---------------:|----------------:|-----------------:|------------------:|
|       1 |      9.17209 |     31.8507 |    2.3792  |   34.2084 |       0.52308  |        0.51984  |       661.048    |          2.02414  |
|       2 |     51.139   |     40.4696 |    3.02349 |   21.9054 |      32.3748   |        0.626387 |      1698.81     |          9.96249  |
|       3 |      4.03087 |     32.9182 |    2.61392 |   34.9776 |       0.824071 |        0.549257 |       117.276    |         11.788    |
|       4 |      2.54472 |     47.0371 |    2.2965  |   45.9679 |       0.54734  |        0.544917 |        43.4443   |          4.2229   |
|       5 |      1.84331 |     54.9946 |    3.85277 |   13.5294 |       0.529979 |        0.714134 |         9.57654  |         32.284    |
|       6 |      1.65032 |     57.8117 |    1.95382 |   50.4133 |       0.525846 |        0.522485 |         2.49132  |          2.24243  |
|       7 |      1.46182 |     63.2298 |    1.92873 |   51.259  |       0.518084 |        0.523533 |         2.56247  |          1.40847  |
|       8 |      1.37961 |     65.615  |    1.87603 |   55.1598 |       0.514352 |        0.519289 |         1.4201   |          3.54838  |
|       9 |      1.33691 |     67.0804 |    1.95435 |   53.3225 |       0.512061 |        0.516759 |         1.50277  |          1.60718  |
|      10 |      1.33067 |     67.8297 |    1.90083 |   55.9002 |       0.513147 |        0.530177 |         2.64856  |          1.62543  |
|      11 |      1.29279 |     68.6946 |    1.91042 |   55.4752 |       0.513878 |        0.517076 |         1.23865  |          1.16685  |
|      12 |      1.25108 |     69.8658 |    2.02394 |   52.5991 |       0.509255 |        0.51509  |         0.797442 |          1.24206  |
|      13 |      1.24042 |     70.2606 |    1.95154 |   56.0217 |       0.508796 |        0.514268 |         0.831014 |          0.959021 |
|      14 |      1.23325 |     70.5389 |    2.08892 |   53.0562 |       0.508526 |        0.514319 |         0.862218 |          1.0034   |
|      15 |      1.2128  |     71.1856 |    2.01233 |   54.27   |       0.507672 |        0.513511 |         0.620296 |          0.939479 |

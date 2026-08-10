# 🔬 Reporte Científico de Experimento: 2026-08-10_13-02-28

## 1. Resumen de Ejecución
- **Duración Total:** 12.97 minutos (778.1 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-06
- **Test Loss:** 31.96194704132614
- **Test Accuracy_Percent:** 9.940308583685486

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
| `MODEL` | `PureCNN2DBaseline_Weighted_9Class` |
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
|   epoch |   train_loss |   train_acc |   val_loss |   val_acc |
|--------:|-------------:|------------:|-----------:|----------:|
|       1 |     1.41865  |     48.7233 |    3.03423 |   29.2182 |
|       2 |     1.21279  |     58.7691 |    2.83373 |   34.0509 |
|       3 |     1.14777  |     61.3792 |    3.38038 |   33.8388 |
|       4 |     1.11041  |     63.0946 |    3.87071 |   24.8975 |
|       5 |     1.08574  |     64.1349 |    4.38774 |   27.041  |
|       6 |     1.03916  |     66.0019 |    5.17649 |   25.9013 |
|       7 |     1.02411  |     66.4369 |    5.86916 |   26.0092 |
|       8 |     1.01305  |     66.9636 |    5.59688 |   28.5594 |
|       9 |     0.986513 |     67.8115 |    6.81116 |   23.7859 |
|      10 |     0.97593  |     68.193  |    6.49524 |   26.1159 |
|      11 |     0.96871  |     68.5127 |    6.90279 |   25.0547 |
|      12 |     0.948291 |     69.431  |    6.61705 |   23.6948 |
|      13 |     0.939111 |     69.7913 |    7.74675 |   24.3492 |
|      14 |     0.931143 |     70.1775 |    7.34552 |   23.4234 |
|      15 |     0.916397 |     70.6631 |    7.21998 |   24.1365 |

# 🔬 Reporte Científico de Experimento: 2026-08-10_11-44-04

## 1. Resumen de Ejecución
- **Duración Total:** 12.97 minutos (777.9 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-06
- **Test Loss:** 13.445314550020102
- **Test Accuracy_Percent:** 30.877596277511678

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
| `NUM_CLASSES` | `3` |
| `BATCH_SIZE` | `256` |
| `NUM_WORKERS` | `4` |
| `EPOCHS` | `15` |
| `LEARNING_RATE` | `0.001` |
| `MODEL` | `PureCNN2DBaseline_Weighted_3Class` |
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
|       1 |     0.525103 |     74.2307 |   0.598387 |   75.0575 |
|       2 |     0.413976 |     80.7063 |   0.730301 |   73.7082 |
|       3 |     0.379778 |     82.8949 |   0.557634 |   77.4817 |
|       4 |     0.366677 |     83.94   |   0.659155 |   76.274  |
|       5 |     0.350204 |     84.7563 |   0.878959 |   75.1755 |
|       6 |     0.341535 |     85.2923 |   0.869136 |   75.7406 |
|       7 |     0.317397 |     86.4156 |   0.837023 |   76.7107 |
|       8 |     0.309968 |     86.7674 |   0.923484 |   75.1767 |
|       9 |     0.301824 |     87.1564 |   0.987256 |   73.0544 |
|      10 |     0.286291 |     87.9082 |   1.07133  |   74.5903 |
|      11 |     0.279763 |     88.1957 |   1.06642  |   75.5036 |
|      12 |     0.274969 |     88.4339 |   1.03725  |   75.7856 |
|      13 |     0.265316 |     88.8877 |   1.03134  |   76.1405 |
|      14 |     0.261901 |     89.0514 |   1.12174  |   75.4306 |
|      15 |     0.258811 |     89.1803 |   1.11974  |   75.2871 |

# 🔬 Reporte Científico de Experimento: 2026-08-18_12-16-06

## 1. Resumen de Ejecución
- **Duración Total:** 39.62 minutos (2377.1 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-01B
- **Test Loss:** 17.19713255465441
- **Test Accuracy_Percent:** 15.85063242363679

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
| `NUM_WORKERS` | `4` |
| `EPOCHS` | `15` |
| `LEARNING_RATE` | `0.001` |
| `MODEL` | `PureCNN2DBaseline_Weighted_5Class` |
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
|   epoch |   train_loss |   train_acc |   val_loss |   val_acc |
|--------:|-------------:|------------:|-----------:|----------:|
|       1 |     1.1006   |     57.9566 |    1.5968  |   44.3612 |
|       2 |     0.939457 |     64.8375 |    2.00588 |   44.7019 |
|       3 |     0.883812 |     67.1878 |    2.33224 |   45.283  |
|       4 |     0.845748 |     68.8055 |    2.46908 |   43.508  |
|       5 |     0.796139 |     70.8262 |    2.52082 |   47.0456 |
|       6 |     0.772407 |     71.7649 |    2.5539  |   47.4436 |
|       7 |     0.753602 |     72.5719 |    3.16743 |   43.1483 |
|       8 |     0.725759 |     73.7186 |    3.4178  |   42.438  |
|       9 |     0.716153 |     74.1074 |    3.47605 |   43.9899 |
|      10 |     0.707025 |     74.4996 |    3.29112 |   43.604  |
|      11 |     0.69039  |     75.1636 |    3.54452 |   44.2178 |
|      12 |     0.683836 |     75.4481 |    3.64104 |   43.881  |
|      13 |     0.678121 |     75.6979 |    3.67711 |   43.7447 |
|      14 |     0.667917 |     76.128  |    3.76299 |   44.4612 |
|      15 |     0.665275 |     76.2552 |    3.79759 |   43.8259 |

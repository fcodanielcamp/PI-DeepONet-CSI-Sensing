# 🔬 Reporte Científico de Experimento: 2026-08-12_02-07-33

## 1. Resumen de Ejecución
- **Duración Total:** 42.42 minutos (2545.1 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-06
- **Test Loss:** 20.477864365156428
- **Test Accuracy_Percent:** 10.481896301782717

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
| `metadata` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/dataset_mc1_rx1_80mhz_meta.npz` | `39ee2e7ece7c366e...` |
| `X_train` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_train_frames.dat` | `4b6d73f1d7ab3d55...` |
| `X_val` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_val_frames.dat` | `3a485662986e2f80...` |
| `X_test` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_test_frames.dat` | `f2cbc80908d965ed...` |

---
## 5. Historial de Entrenamiento por Época
|   epoch |   train_loss |   train_acc |   val_loss |   val_acc |
|--------:|-------------:|------------:|-----------:|----------:|
|       1 |     1.43858  |     49.9167 |    2.29566 |   29.9478 |
|       2 |     1.24732  |     58.1109 |    2.33824 |   34.9811 |
|       3 |     1.18542  |     60.6622 |    2.14953 |   38.0943 |
|       4 |     1.14837  |     62.0361 |    2.2857  |   36.8885 |
|       5 |     1.12131  |     63.0553 |    2.3638  |   40.0817 |
|       6 |     1.094    |     63.9659 |    2.41626 |   39.4574 |
|       7 |     1.04481  |     65.7701 |    3.04612 |   35.7529 |
|       8 |     1.03039  |     66.2839 |    3.22668 |   35.3164 |
|       9 |     1.01786  |     66.633  |    3.16407 |   36.9585 |
|      10 |     0.989785 |     67.584  |    3.37791 |   35.7163 |
|      11 |     0.981379 |     67.8716 |    3.47171 |   35.6105 |
|      12 |     0.973428 |     68.0876 |    3.84646 |   35.187  |
|      13 |     0.956421 |     68.5749 |    3.6327  |   35.8663 |
|      14 |     0.951596 |     68.7557 |    3.41449 |   36.6832 |
|      15 |     0.946611 |     68.9116 |    3.63122 |   36.1908 |

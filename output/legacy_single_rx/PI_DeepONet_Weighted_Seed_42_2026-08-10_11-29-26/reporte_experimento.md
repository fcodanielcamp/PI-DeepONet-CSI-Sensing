# 🔬 Reporte Científico de Experimento: 2026-08-10_11-29-26

## 1. Resumen de Ejecución
- **Duración Total:** 13.91 minutos (834.9 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-06
- **Test Loss:** 6.968888432556753
- **Test Accuracy_Percent:** 30.87289616599085

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
| `LAMBDA_PHYS` | `0.01` |
| `MODEL` | `PI-DeepONet_Weighted_3Class` |
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
|       1 |     0.548521 |     72.5632 |   0.760698 |   64.6527 |
|       2 |     0.408496 |     80.4044 |   0.636441 |   73.5697 |
|       3 |     0.37797  |     82.1877 |   0.725202 |   69.5467 |
|       4 |     0.36096  |     83.1437 |   0.701615 |   73.7856 |
|       5 |     0.349123 |     83.8049 |   0.739267 |   72.614  |
|       6 |     0.329025 |     84.8877 |   0.737148 |   74.5036 |
|       7 |     0.322107 |     85.2579 |   0.887906 |   72.4181 |
|       8 |     0.316336 |     85.6062 |   0.680064 |   76.5161 |
|       9 |     0.304637 |     86.2436 |   0.776491 |   74.8018 |
|      10 |     0.301028 |     86.4524 |   0.835304 |   73.8124 |
|      11 |     0.298499 |     86.5658 |   0.81347  |   74.5117 |
|      12 |     0.290818 |     86.9535 |   0.866284 |   74.7357 |
|      13 |     0.288854 |     87.0699 |   0.835591 |   74.8966 |
|      14 |     0.286782 |     87.1797 |   0.876556 |   74.3701 |
|      15 |     0.282422 |     87.4058 |   0.899474 |   74.1012 |

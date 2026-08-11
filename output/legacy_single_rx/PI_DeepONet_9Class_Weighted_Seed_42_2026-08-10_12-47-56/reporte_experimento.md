# 🔬 Reporte Científico de Experimento: 2026-08-10_12-47-56

## 1. Resumen de Ejecución
- **Duración Total:** 13.78 minutos (826.9 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-06
- **Test Loss:** 19.534223435364016
- **Test Accuracy_Percent:** 10.00141003345625

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
| `MODEL` | `PI-DeepONet_Weighted_9Class` |
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
|       1 |     1.38353  |     49.9178 |    2.14011 |   32.2069 |
|       2 |     1.09812  |     63.0443 |    2.6776  |   21.8851 |
|       3 |     1.0267   |     66.0107 |    2.69946 |   25.517  |
|       4 |     0.980743 |     67.8515 |    3.45863 |   21.2675 |
|       5 |     0.921181 |     70.0246 |    3.5908  |   24.6723 |
|       6 |     0.900669 |     70.7041 |    3.2971  |   27.7528 |
|       7 |     0.881589 |     71.3313 |    4.15419 |   22.9712 |
|       8 |     0.844298 |     72.5647 |    5.25033 |   21.6599 |
|       9 |     0.830024 |     73.0226 |    4.84821 |   22.2506 |
|      10 |     0.816858 |     73.4091 |    4.39639 |   24.1583 |
|      11 |     0.792235 |     74.1413 |    5.12245 |   22.7684 |
|      12 |     0.782601 |     74.4148 |    5.31047 |   23.2712 |
|      13 |     0.775055 |     74.6749 |    5.20116 |   23.5919 |
|      14 |     0.759031 |     75.1857 |    5.53161 |   23.7141 |
|      15 |     0.753794 |     75.2555 |    5.45938 |   23.2325 |

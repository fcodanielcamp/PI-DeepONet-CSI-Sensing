# 🔬 Reporte Científico de Experimento: 2026-08-11_15-43-04

## 1. Resumen de Ejecución
- **Duración Total:** 45.75 minutos (2744.9 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-06
- **Test Loss:** 10.859020060562646
- **Test Accuracy_Percent:** 10.334737514039777

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
| `metadata` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/dataset_mc1_rx1_80mhz_meta.npz` | `39ee2e7ece7c366e...` |
| `X_train` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_train_frames.dat` | `4b6d73f1d7ab3d55...` |
| `X_val` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_val_frames.dat` | `3a485662986e2f80...` |
| `X_test` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_test_frames.dat` | `f2cbc80908d965ed...` |

---
## 5. Historial de Entrenamiento por Época
|   epoch |   train_loss |   train_acc |   val_loss |   val_acc |   tr_loss_temp |   val_loss_temp |
|--------:|-------------:|------------:|-----------:|----------:|---------------:|----------------:|
|       1 |     1.35924  |     51.5894 |    1.6196  |   36.9945 |        6.2e-05 |         2.3e-05 |
|       2 |     1.14412  |     61.0269 |    1.79983 |   37.8213 |        3e-05   |         2.6e-05 |
|       3 |     1.07908  |     63.5313 |    2.11574 |   35.75   |        1.7e-05 |         1e-05   |
|       4 |     1.03596  |     65.0841 |    2.02913 |   37.4262 |        1.1e-05 |         1.2e-05 |
|       5 |     0.973527 |     67.1783 |    2.40396 |   34.6864 |        4e-06   |         3e-06   |
|       6 |     0.951966 |     67.8614 |    2.39757 |   34.8799 |        3e-06   |         4e-06   |
|       7 |     0.934008 |     68.3586 |    2.36012 |   34.9556 |        2e-06   |         4e-06   |
|       8 |     0.897131 |     69.4759 |    2.46691 |   36.866  |        2e-06   |         1e-06   |
|       9 |     0.885731 |     69.8337 |    2.75638 |   34.3349 |        1e-06   |         1e-06   |
|      10 |     0.875658 |     70.0508 |    2.61198 |   35.6891 |        1e-06   |         1e-06   |
|      11 |     0.854497 |     70.663  |    2.80791 |   34.8521 |        1e-06   |         1e-06   |
|      12 |     0.847503 |     70.8495 |    2.68778 |   36.0488 |        1e-06   |         1e-06   |
|      13 |     0.842197 |     71.0232 |    3.10892 |   33.5395 |        1e-06   |         1e-06   |
|      14 |     0.82962  |     71.3678 |    3.0533  |   34.445  |        1e-06   |         1e-06   |
|      15 |     0.826092 |     71.4904 |    2.95595 |   34.997  |        1e-06   |         1e-06   |

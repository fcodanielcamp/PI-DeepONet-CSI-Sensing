# 🔬 Reporte Científico de Experimento: 2026-08-12_12-57-45

## 1. Resumen de Ejecución
- **Duración Total:** 46.87 minutos (2812.0 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-06
- **Test Loss:** 7.814626024024509
- **Test Accuracy_Percent:** 10.847478687558016

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
| `LAMBDA_PHYS` | `0.0` |
| `LAMBDA_TEMP` | `0.0` |
| `LAMBDA_ENERGY` | `0.1` |
| `GAMMA` | `1.0` |
| `MODEL` | `PI-DeepONet_CORAL_Ordinal_Energy` |
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
|   epoch |   train_loss |   train_acc |   val_loss |   val_acc |   tr_loss_temp |   val_loss_temp |   tr_loss_energy |   val_loss_energy |
|--------:|-------------:|------------:|-----------:|----------:|---------------:|----------------:|-----------------:|------------------:|
|       1 |     68.4964  |     19.5885 |    3.25266 |   18.5079 |        2.1e-05 |         7e-06   |       651.176    |          2.60744  |
|       2 |      2.54203 |     37.3205 |    2.12199 |   37.4625 |        9e-06   |         1.1e-05 |         2.07283  |          1.51511  |
|       3 |     34.9161  |     30.8607 |    2.51593 |   38.7802 |        1.7e-05 |         6e-06   |       322.011    |          2.63275  |
|       4 |      3.35864 |     41.4873 |    2.37013 |   38.3584 |        9e-06   |         5e-06   |        12.0448   |          1.96565  |
|       5 |      1.99635 |     47.0369 |    1.94061 |   46.6929 |        7e-06   |         9e-06   |         1.53188  |          1.50464  |
|       6 |      5.0065  |     41.4313 |    1.8957  |   46.0466 |        9e-06   |         9e-06   |        28.7416   |          1.50488  |
|       7 |      1.6175  |     52.5606 |    1.57098 |   52.0253 |        1e-05   |         8e-06   |         1.29447  |          1.27479  |
|       8 |      1.43333 |     55.7199 |    1.52397 |   51.8163 |        1.2e-05 |         1.1e-05 |         1.14965  |          1.23883  |
|       9 |      1.34365 |     57.5585 |    1.45449 |   52.1456 |        1.3e-05 |         1.1e-05 |         1.03841  |          1.05539  |
|      10 |      1.29496 |     58.6045 |    1.49692 |   51.4327 |        1.4e-05 |         1.2e-05 |         0.980167 |          1.1977   |
|      11 |      1.26154 |     59.3736 |    1.52967 |   49.648  |        1.5e-05 |         1.4e-05 |         0.942346 |          1.09011  |
|      12 |      1.23489 |     60.0021 |    1.46462 |   50.5033 |        1.6e-05 |         1.5e-05 |         0.91164  |          0.968985 |
|      13 |      1.17877 |     61.2021 |    1.48432 |   51.2196 |        1.7e-05 |         1.5e-05 |         0.786039 |          0.952077 |
|      14 |      1.16447 |     61.5974 |    1.47217 |   51.0687 |        1.7e-05 |         1.4e-05 |         0.781971 |          0.97145  |
|      15 |      1.15564 |     61.8605 |    1.48843 |   50.608  |        1.7e-05 |         1.4e-05 |         0.778955 |          0.95795  |

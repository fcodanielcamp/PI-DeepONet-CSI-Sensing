# 🔬 Reporte Científico de Experimento: 2026-08-12_08-59-19

## 1. Resumen de Ejecución
- **Duración Total:** 46.39 minutos (2783.6 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-06
- **Test Loss:** 6.490549507706221
- **Test Accuracy_Percent:** 11.132659748904228

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
| `LAMBDA_TEMP` | `0.0` |
| `LAMBDA_ENERGY` | `0.01` |
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
|       1 |     10.7165  |     29.0605 |    2.89869 |   22.8139 |        2.4e-05 |         1.1e-05 |       769.971    |           3.28237 |
|       2 |      2.21161 |     40.6969 |    1.82975 |   47.2259 |        7e-06   |         7e-06   |         3.46859  |           3.2689  |
|       3 |      2.50293 |     42.5536 |    1.78522 |   44.6961 |        9e-06   |         6e-06   |        48.3257   |           2.48197 |
|       4 |      1.84898 |     48.9744 |    1.67381 |   47.741  |        1e-05   |         4e-06   |        20.6307   |           3.2075  |
|       5 |      1.43228 |     53.3999 |    1.62188 |   47.907  |        5e-06   |         7e-06   |         2.68241  |           3.79283 |
|       6 |      1.30621 |     56.2301 |    1.48905 |   50.7409 |        9e-06   |         9e-06   |         2.29681  |           2.26092 |
|       7 |      1.23652 |     57.9089 |    1.45229 |   50.5759 |        1e-05   |         1.2e-05 |         2.14261  |           2.01307 |
|       8 |      1.19107 |     59.07   |    1.49368 |   47.6466 |        1.2e-05 |         1e-05   |         2.03776  |           1.6332  |
|       9 |      1.16065 |     59.8508 |    1.54736 |   46.4044 |        1.2e-05 |         9e-06   |         1.97164  |           1.90616 |
|      10 |      1.13603 |     60.5268 |    1.50359 |   48.9756 |        1.3e-05 |         1e-05   |         1.90546  |           1.9208  |
|      11 |      1.0869  |     61.7162 |    1.59947 |   45.6394 |        1.5e-05 |         1.3e-05 |         1.23098  |           1.30432 |
|      12 |      1.07422 |     62.1401 |    1.4841  |   50.2652 |        1.6e-05 |         1.1e-05 |         1.24655  |           1.63974 |
|      13 |      1.06499 |     62.4181 |    1.59572 |   47.0088 |        1.6e-05 |         1.3e-05 |         1.24341  |           1.53974 |
|      14 |      1.03897 |     63.083  |    1.55687 |   46.3901 |        1.7e-05 |         1.4e-05 |         0.955614 |           1.11309 |
|      15 |      1.03345 |     63.2607 |    1.59009 |   46.1889 |        1.7e-05 |         1.4e-05 |         0.955786 |           1.08525 |

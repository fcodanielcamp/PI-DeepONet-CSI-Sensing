# 🔬 Reporte Científico de Experimento: 2026-08-12_20-14-07

## 1. Resumen de Ejecución
- **Duración Total:** 45.92 minutos (2755.0 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-06
- **Test Loss:** 8.282186257813093
- **Test Accuracy_Percent:** 11.152029158880326

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
| `EPOCHS` | `15` |
| `LEARNING_RATE` | `0.001` |
| `LAMBDA_ENERGY` | `0.1` |
| `LAMBDA_MONO` | `0.001` |
| `MARGIN` | `0.0` |
| `WARMUP_EPOCHS` | `5` |
| `MIX_RATIO` | `0.0` |
| `GAMMA` | `1.0` |
| `MODEL` | `PI-DeepONet_CORAL_Monotonic_Energy` |
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
|   epoch |   train_loss |   train_acc |   val_loss |   val_acc |   tr_loss_energy |   val_loss_energy |   tr_loss_mono |   val_loss_mono |
|--------:|-------------:|------------:|-----------:|----------:|-----------------:|------------------:|---------------:|----------------:|
|       1 |     70.2066  |     19.3798 |    3.39585 |   14.3993 |       668.12     |          2.66619  |       0        |        0        |
|       2 |      2.75064 |     33.7758 |    2.25289 |   36.5032 |         2.33242  |          1.58482  |       0        |        0        |
|       3 |     14.7909  |     30.4392 |    2.45176 |   37.1982 |       121.021    |          1.87914  |       0        |        0        |
|       4 |      3.13498 |     45.7019 |    5.41158 |   13.3881 |        12.401    |         13.6323   |       0        |        0        |
|       5 |      2.09173 |     44.5776 |    1.81586 |   48.405  |         1.81486  |          1.68761  |       0        |        0        |
|       6 |      2.79249 |     47.1331 |    1.71183 |   49.4465 |        10.2543   |          1.43894  |       0.050824 |        0.001586 |
|       7 |      1.50703 |     54.0442 |    1.55285 |   51.2428 |         1.21861  |          1.17507  |       0.040115 |        0.000924 |
|       8 |      1.38789 |     56.3649 |    1.51008 |   52.4711 |         1.06761  |          1.18133  |       0.037154 |        0.001495 |
|       9 |      1.32656 |     57.7091 |    1.51518 |   50.2959 |         0.989426 |          1.03866  |       0.034776 |        0.001253 |
|      10 |      1.28658 |     58.6459 |    1.54406 |   49.3732 |         0.947436 |          1.07216  |       0.033565 |        0.001096 |
|      11 |      1.25642 |     59.3411 |    1.59243 |   47.8772 |         0.90864  |          1.09632  |       0.032703 |        0.001069 |
|      12 |      1.19836 |     60.5401 |    1.46722 |   52.1325 |         0.77939  |          0.882162 |       0.030165 |        0.001104 |
|      13 |      1.1854  |     60.9383 |    1.5179  |   50.8688 |         0.776555 |          0.902971 |       0.02982  |        0.000889 |
|      14 |      1.17475 |     61.1899 |    1.52457 |   50.0098 |         0.768717 |          0.974847 |       0.029161 |        0.000705 |
|      15 |      1.16649 |     61.3986 |    1.51421 |   50.4492 |         0.765055 |          0.936155 |       0.029105 |        0.001079 |

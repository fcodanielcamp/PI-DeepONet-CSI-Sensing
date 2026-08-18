# 🔬 Reporte Científico de Experimento: 2026-08-18_10-46-46

## 1. Resumen de Ejecución
- **Duración Total:** 43.13 minutos (2587.8 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-01B
- **Test Loss:** 5.6487962563484215
- **Test Accuracy_Percent:** 16.249422589907685

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
| `EPOCHS` | `15` |
| `LEARNING_RATE` | `0.001` |
| `LAMBDA_ENERGY` | `0.01` |
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
| `metadata` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/dataset_mc1_rx1_80mhz_meta.npz` | `b4779fcd95fefadd...` |
| `X_train` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_train_frames.dat` | `17b8567418dad173...` |
| `X_val` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_val_frames.dat` | `a323d19d9f0be89a...` |
| `X_test` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_test_frames.dat` | `5a03477aa2b0d325...` |

---
## 5. Historial de Entrenamiento por Época
|   epoch |   train_loss |   train_acc |   val_loss |   val_acc |   tr_loss_energy |   val_loss_energy |   tr_loss_mono |   val_loss_mono |
|--------:|-------------:|------------:|-----------:|----------:|-----------------:|------------------:|---------------:|----------------:|
|       1 |     8.69388  |     32.6639 |    1.76277 |   42.4036 |       667.487    |          2.00097  |       0        |        0        |
|       2 |     5.70877  |     47.5994 |    2.09437 |   36.986  |       421.598    |          2.03301  |       0        |        0        |
|       3 |     1.51466  |     48.9847 |    1.62277 |   46.6447 |         2.86775  |          4.88096  |       0        |        0        |
|       4 |     2.03546  |     53.439  |    1.82697 |   41.3686 |        75.2731   |          2.25851  |       0        |        0        |
|       5 |     1.23088  |     55.4664 |    1.29735 |   52.0524 |         2.69039  |          1.98568  |       0        |        0        |
|       6 |     0.969844 |     62.0758 |    1.44748 |   49.0809 |         1.86053  |          2.75585  |       0.056597 |        0.001933 |
|       7 |     0.888868 |     64.8807 |    1.35886 |   53.3051 |         1.71667  |          2.73307  |       0.054131 |        0.003571 |
|       8 |     0.845442 |     66.4169 |    1.46968 |   50.9456 |         1.55663  |          1.92206  |       0.050512 |        0.004068 |
|       9 |     0.793409 |     68.0422 |    1.3654  |   54.0766 |         0.978148 |          1.30058  |       0.037368 |        0.002094 |
|      10 |     0.778438 |     68.5524 |    1.41461 |   52.8187 |         0.993043 |          1.39202  |       0.036858 |        0.002257 |
|      11 |     0.76785  |     68.9526 |    1.49061 |   51.344  |         0.994948 |          1.31475  |       0.036496 |        0.002023 |
|      12 |     0.743063 |     69.7483 |    1.42935 |   54.3165 |         0.700784 |          1.10141  |       0.028969 |        0.001614 |
|      13 |     0.737222 |     69.9706 |    1.45088 |   53.5023 |         0.715687 |          1.06289  |       0.029551 |        0.001509 |
|      14 |     0.732127 |     70.1863 |    1.48962 |   52.326  |         0.721723 |          0.967051 |       0.028388 |        0.001239 |
|      15 |     0.718764 |     70.6214 |    1.43386 |   54.2437 |         0.578647 |          0.849522 |       0.024927 |        0.001385 |

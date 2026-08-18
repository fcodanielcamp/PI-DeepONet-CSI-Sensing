# 🔬 Reporte Científico de Experimento: 2026-08-12_18-36-44

## 1. Resumen de Ejecución
- **Duración Total:** 46.56 minutos (2793.7 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-06
- **Test Loss:** 5.974084140293986
- **Test Accuracy_Percent:** 11.026188904129947

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
| `LAMBDA_ENERGY` | `0.01` |
| `LAMBDA_MONO` | `0.001` |
| `MARGIN` | `0.0` |
| `WARMUP_EPOCHS` | `5` |
| `MIX_RATIO` | `0.3` |
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
|       1 |     11.1387  |     28.9944 |    3.06186 |   16.5224 |        812.101   |           3.94031 |       0        |        0        |
|       2 |      2.34859 |     37.9694 |    1.91389 |   45.2762 |          3.98342 |           9.16924 |       0        |        0        |
|       3 |      3.51335 |     41.5335 |    2.00115 |   43.8413 |        143.112   |           2.83155 |       0        |        0        |
|       4 |      3.42567 |     47.4106 |    1.89479 |   45.6099 |        168.984   |           2.37407 |       0        |        0        |
|       5 |      1.52956 |     51.48   |    1.65073 |   46.5379 |          2.39748 |           1.84603 |       0        |        0        |
|       6 |      2.03977 |     50.5419 |    1.61486 |   47.7247 |         47.8656  |           3.73351 |       0.050178 |        0.003487 |
|       7 |      1.34111 |     55.0219 |    1.4877  |   49.6708 |          2.1842  |           1.58683 |       0.041358 |        0.001462 |
|       8 |      1.25224 |     57.1121 |    1.54811 |   46.9413 |          1.85406 |           2.93012 |       0.038484 |        0.002774 |
|       9 |      1.19961 |     58.5968 |    1.52684 |   47.0179 |          1.95037 |           2.14407 |       0.038509 |        0.002054 |
|      10 |      1.16501 |     59.6089 |    1.49237 |   47.4088 |          1.9999  |           2.02561 |       0.038456 |        0.001939 |
|      11 |      1.10982 |     60.9516 |    1.57286 |   46.9781 |          1.32664 |           1.23903 |       0.029968 |        0.000914 |
|      12 |      1.09322 |     61.4119 |    1.51358 |   49.9546 |          1.33723 |           1.51073 |       0.029751 |        0.001153 |
|      13 |      1.08168 |     61.7955 |    1.59436 |   47.4792 |          1.34964 |           1.3083  |       0.029565 |        0.001228 |
|      14 |      1.0531  |     62.4988 |    1.60776 |   46.5224 |          1.00182 |           1.01117 |       0.025088 |        0.00096  |
|      15 |      1.04592 |     62.696  |    1.59615 |   47.2195 |          1.01269 |           1.29126 |       0.024962 |        0.001341 |

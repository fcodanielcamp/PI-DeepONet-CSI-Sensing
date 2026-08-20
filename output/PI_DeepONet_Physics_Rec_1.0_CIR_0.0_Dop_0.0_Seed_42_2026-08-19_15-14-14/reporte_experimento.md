# 🔬 Reporte Científico de Experimento: 2026-08-19_15-14-14

## 1. Resumen de Ejecución
- **Duración Total:** 85.63 minutos (5137.6 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-01B
- **Test Loss:** 8.668016000236312
- **Test Accuracy_Percent:** 15.830437578196996

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
| `LAMBDA_ENERGY` | `0.0` |
| `LAMBDA_MONO` | `0.0` |
| `LAMBDA_REC` | `1.0` |
| `LAMBDA_CIR` | `0.0` |
| `LAMBDA_DOP` | `0.0` |
| `MODEL` | `PI-DeepONet_Physics_Fourier` |
| `DEVICE` | `cuda` |

---
## 4. Trazabilidad de Archivos de Datos (Hashes SHA-256)
| Identificador | Ruta | SHA-256 Hash |
| :--- | :--- | :--- |
| `metadata` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/dataset_mc1_rx1_80mhz_meta.npz` | `532cc6f8e5410fa9...` |
| `X_train` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_train_frames.dat` | `e29dc479c9c23cbb...` |
| `X_val` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_val_frames.dat` | `0513eb4ed4a856a7...` |
| `X_test` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_test_frames.dat` | `e2b1497acf70184b...` |

---
## 5. Historial de Entrenamiento por Época
|   epoch |   train_loss |   train_acc |   val_loss |   val_acc |   tr_loss_phys |   val_loss_phys |   tr_loss_energy |   val_loss_energy |
|--------:|-------------:|------------:|-----------:|----------:|---------------:|----------------:|-----------------:|------------------:|
|       1 |     1.2978   |     60.1014 |    1.43587 |   55.3841 |       0.26959  |        0.263357 |      1.48145e+06 |       1.37046e+06 |
|       2 |     1.02898  |     68.5539 |    1.7623  |   49.4767 |       0.259259 |        0.261676 |      1.74692e+06 |       1.29224e+06 |
|       3 |     0.961936 |     71.0025 |    1.69629 |   54.4728 |       0.257863 |        0.261329 |      1.38374e+06 |       1.42476e+06 |
|       4 |     0.92331  |     72.468  |    1.9392  |   49.8017 |       0.257317 |        0.260042 |      1.17224e+06 |  969122           |
|       5 |     0.873444 |     74.2653 |    1.83539 |   53.3848 |       0.256022 |        0.258957 |      1.25095e+06 |       1.18094e+06 |
|       6 |     0.85747  |     74.9146 |    1.91911 |   53.5488 |       0.255865 |        0.258777 |      1.31822e+06 |       1.45994e+06 |
|       7 |     0.844701 |     75.4229 |    1.95648 |   53.0506 |       0.255797 |        0.258746 |      1.32581e+06 |       1.45602e+06 |
|       8 |     0.819167 |     76.3746 |    1.9941  |   53.435  |       0.255339 |        0.258341 |      1.37774e+06 |       1.33267e+06 |
|       9 |     0.811245 |     76.7054 |    2.04937 |   51.0418 |       0.25525  |        0.258386 |      1.47049e+06 |       1.53372e+06 |
|      10 |     0.804607 |     76.9612 |    2.0614  |   52.2053 |       0.255147 |        0.258125 |      1.47347e+06 |       1.66642e+06 |
|      11 |     0.790411 |     77.4995 |    2.04127 |   52.7317 |       0.254883 |        0.257892 |      1.58909e+06 |       1.56871e+06 |
|      12 |     0.786298 |     77.6393 |    2.06199 |   52.0405 |       0.254847 |        0.257881 |      1.60113e+06 |       1.7977e+06  |
|      13 |     0.782184 |     77.8426 |    2.11959 |   51.2203 |       0.254821 |        0.257907 |      1.66387e+06 |       1.6455e+06  |
|      14 |     0.774212 |     78.1346 |    2.12814 |   51.2531 |       0.254707 |        0.257745 |      1.70892e+06 |       1.71235e+06 |
|      15 |     0.772124 |     78.2265 |    2.11887 |   52.0753 |       0.254688 |        0.257751 |      1.73701e+06 |       1.69662e+06 |

# 🔬 Reporte Científico de Experimento: 2026-08-19_09-13-02

## 1. Resumen de Ejecución
- **Duración Total:** 85.61 minutos (5136.8 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-01B
- **Test Loss:** 10.267496863769622
- **Test Accuracy_Percent:** 15.719946239928692

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
|       1 |     1.34497  |     58.6964 |    1.59683 |   52.1986 |       0.265644 |        0.263522 |      7.41064e+07 |       8.4082e+07  |
|       2 |     1.09702  |     66.1896 |    1.69889 |   51.8864 |       0.25855  |        0.261658 |      4.5434e+07  |       3.30274e+07 |
|       3 |     1.03127  |     68.5432 |    1.73862 |   51.8407 |       0.257847 |        0.264306 |      1.75259e+07 |       1.20228e+07 |
|       4 |     0.994697 |     69.9018 |    1.88969 |   52.7821 |       0.257593 |        0.260564 |      8.944e+06   |       6.22562e+06 |
|       5 |     0.938955 |     71.9212 |    1.78558 |   53.1425 |       0.256044 |        0.259014 |      6.84562e+06 |       6.40008e+06 |
|       6 |     0.921887 |     72.5743 |    1.85331 |   53.7731 |       0.255893 |        0.258925 |      5.54777e+06 |       5.15672e+06 |
|       7 |     0.908885 |     73.0542 |    1.91062 |   52.4175 |       0.255686 |        0.258603 |      4.49248e+06 |       3.8033e+06  |
|       8 |     0.879549 |     74.1531 |    1.93361 |   52.1583 |       0.254906 |        0.257795 |      4.31133e+06 |       4.61846e+06 |
|       9 |     0.871666 |     74.4192 |    2.03808 |   50.747  |       0.254738 |        0.257761 |      4.22939e+06 |       4.30736e+06 |
|      10 |     0.865156 |     74.704  |    1.99738 |   51.164  |       0.254669 |        0.257584 |      4.124e+06   |       4.32782e+06 |
|      11 |     0.849517 |     75.3047 |    1.91932 |   53.0583 |       0.2543   |        0.257285 |      4.26955e+06 |       4.61242e+06 |
|      12 |     0.845168 |     75.4608 |    2.01758 |   52.2019 |       0.254198 |        0.257179 |      4.45512e+06 |       4.56528e+06 |
|      13 |     0.840881 |     75.6378 |    1.99074 |   52.4495 |       0.254141 |        0.257133 |      4.44505e+06 |       4.61271e+06 |
|      14 |     0.832484 |     75.948  |    2.02106 |   51.5825 |       0.253955 |        0.256941 |      4.62617e+06 |       5.05724e+06 |
|      15 |     0.83012  |     76.0445 |    2.02502 |   52.3982 |       0.253922 |        0.256976 |      4.76685e+06 |       5.17148e+06 |

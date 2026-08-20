# 🔬 Reporte Científico de Experimento: 2026-08-19_13-45-02

## 1. Resumen de Ejecución
- **Duración Total:** 85.58 minutos (5134.6 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-01B
- **Test Loss:** 8.876268085756308
- **Test Accuracy_Percent:** 15.78772664071513

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
| `LAMBDA_REC` | `0.0` |
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
|       1 |     1.02843  |     59.9964 |    1.33533 |   55.1472 |              0 |               0 |      1.11623e+06 |       1.01168e+06 |
|       2 |     0.770888 |     68.3775 |    1.62981 |   48.5705 |              0 |               0 |      1.20979e+06 |  993604           |
|       3 |     0.704338 |     70.9861 |    1.4338  |   55.5746 |              0 |               0 | 964002           |  758203           |
|       4 |     0.665256 |     72.4537 |    1.61612 |   50.8306 |              0 |               0 | 836553           |  610213           |
|       5 |     0.616918 |     74.2918 |    1.57628 |   53.7683 |              0 |               0 | 728882           |  535722           |
|       6 |     0.599856 |     75.0152 |    1.68546 |   52.1389 |              0 |               0 | 697382           |  558491           |
|       7 |     0.587015 |     75.4899 |    1.65145 |   53.3168 |              0 |               0 | 706429           |  673379           |
|       8 |     0.561605 |     76.4621 |    1.7067  |   53.3399 |              0 |               0 | 709179           |  605685           |
|       9 |     0.552882 |     76.8341 |    1.80018 |   50.9452 |              0 |               0 | 743130           |  626702           |
|      10 |     0.546121 |     77.1189 |    1.82129 |   51.7943 |              0 |               0 | 760824           |  715672           |
|      11 |     0.53189  |     77.691  |    1.7996  |   52.6088 |              0 |               0 | 777303           |  724164           |
|      12 |     0.527529 |     77.7957 |    1.89658 |   50.8308 |              0 |               0 | 802282           |  717143           |
|      13 |     0.523633 |     77.9865 |    1.94383 |   50.2732 |              0 |               0 | 800786           |  665165           |
|      14 |     0.515863 |     78.324  |    1.93551 |   51.0122 |              0 |               0 | 823299           |  758501           |
|      15 |     0.513335 |     78.4282 |    1.95257 |   50.9594 |              0 |               0 | 845391           |  754812           |

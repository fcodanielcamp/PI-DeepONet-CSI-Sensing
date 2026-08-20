# 🔬 Reporte Científico de Experimento: 2026-08-19_22-43-03

## 1. Resumen de Ejecución
- **Duración Total:** 85.63 minutos (5137.7 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-01B
- **Test Loss:** 7.901086975822405
- **Test Accuracy_Percent:** 15.726213605754836

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
| `SEED` | `777` |
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
|       1 |     1.03773  |     59.9614 |    1.16026 |   53.4607 |              0 |               0 |      1.53717e+06 |       1.82912e+06 |
|       2 |     0.76817  |     68.6421 |    1.27136 |   53.8185 |              0 |               0 |      1.28751e+06 |       1.11963e+06 |
|       3 |     0.705558 |     70.9859 |    1.42256 |   53.2287 |              0 |               0 |      1.04419e+06 |       1.4855e+06  |
|       4 |     0.668968 |     72.349  |    1.48071 |   51.997  |              0 |               0 | 888059           |       1.02161e+06 |
|       5 |     0.624037 |     74.0921 |    1.50957 |   53.0074 |              0 |               0 | 922194           |  878280           |
|       6 |     0.608583 |     74.6849 |    1.63837 |   50.473  |              0 |               0 | 912913           |       1.09459e+06 |
|       7 |     0.596073 |     75.1311 |    1.63825 |   51.2284 |              0 |               0 | 927518           |  977197           |
|       8 |     0.572394 |     76.0842 |    1.66551 |   51.4987 |              0 |               0 | 993121           |       1.00451e+06 |
|       9 |     0.564904 |     76.3929 |    1.68697 |   52.1531 |              0 |               0 |      1.02389e+06 |       1.0524e+06  |
|      10 |     0.558335 |     76.638  |    1.77672 |   51.5271 |              0 |               0 |      1.05001e+06 |  970975           |
|      11 |     0.545158 |     77.1508 |    1.7635  |   51.5374 |              0 |               0 |      1.10762e+06 |       1.09127e+06 |
|      12 |     0.541252 |     77.2822 |    1.74847 |   53.1083 |              0 |               0 |      1.14749e+06 |       1.12395e+06 |
|      13 |     0.537657 |     77.4483 |    1.8786  |   49.9865 |              0 |               0 |      1.16627e+06 |       1.13712e+06 |
|      14 |     0.530331 |     77.7316 |    1.85754 |   51.4878 |              0 |               0 |      1.23079e+06 |       1.21092e+06 |
|      15 |     0.528204 |     77.8155 |    1.82982 |   51.7194 |              0 |               0 |      1.23086e+06 |       1.25461e+06 |

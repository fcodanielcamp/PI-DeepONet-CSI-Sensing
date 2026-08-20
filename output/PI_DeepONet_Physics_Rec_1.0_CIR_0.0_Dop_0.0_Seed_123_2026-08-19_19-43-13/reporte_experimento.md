# 🔬 Reporte Científico de Experimento: 2026-08-19_19-43-13

## 1. Resumen de Ejecución
- **Duración Total:** 85.72 minutos (5143.4 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-01B
- **Test Loss:** 8.624049843876659
- **Test Accuracy_Percent:** 15.857828288103843

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
| `SEED` | `123` |
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
|       1 |     1.29392  |     60.2734 |    1.55828 |   52.6515 |       0.26976  |        0.264771 |      1.22611e+06 |            873274 |
|       2 |     1.02764  |     68.5679 |    1.74153 |   51.5451 |       0.259433 |        0.261956 | 858789           |            482756 |
|       3 |     0.961915 |     71.014  |    1.72412 |   53.2324 |       0.25797  |        0.26068  | 574887           |            437702 |
|       4 |     0.923497 |     72.4348 |    1.82666 |   50.8547 |       0.257351 |        0.260378 | 444064           |            341435 |
|       5 |     0.874845 |     74.2452 |    1.81853 |   52.9923 |       0.25628  |        0.25907  | 406407           |            372441 |
|       6 |     0.859388 |     74.8395 |    1.86797 |   52.656  |       0.255992 |        0.2589   | 397999           |            375702 |
|       7 |     0.846635 |     75.3198 |    1.96772 |   51.7694 |       0.255841 |        0.25882  | 401432           |            374245 |
|       8 |     0.821932 |     76.309  |    2.02303 |   51.2612 |       0.255379 |        0.258354 | 399699           |            407069 |
|       9 |     0.813898 |     76.6164 |    2.01963 |   51.8004 |       0.255279 |        0.258227 | 424245           |            434887 |
|      10 |     0.807765 |     76.8518 |    2.03051 |   52.2108 |       0.255193 |        0.258157 | 423253           |            429938 |
|      11 |     0.792996 |     77.4472 |    2.07507 |   51.8307 |       0.254931 |        0.257942 | 441693           |            472583 |
|      12 |     0.789026 |     77.5945 |    2.10187 |   51.9016 |       0.25489  |        0.257919 | 447917           |            455370 |
|      13 |     0.785411 |     77.7355 |    2.13978 |   51.471  |       0.254869 |        0.257892 | 465670           |            478856 |
|      14 |     0.777292 |     78.0467 |    2.14549 |   51.5852 |       0.254742 |        0.257779 | 468844           |            461310 |
|      15 |     0.775387 |     78.1121 |    2.12718 |   52.326  |       0.254722 |        0.257769 | 472107           |            471786 |

# 🔬 Reporte Científico de Experimento: 2026-08-19_10-41-50

## 1. Resumen de Ejecución
- **Duración Total:** 86.63 minutos (5197.5 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-01B
- **Test Loss:** 6.2916276059340515
- **Test Accuracy_Percent:** 15.905645968110715

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
| `LAMBDA_REC` | `1.0` |
| `LAMBDA_CIR` | `0.01` |
| `LAMBDA_DOP` | `0.01` |
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
|       1 |     62.3262  |     24.9049 |    2.85068 |   25.0149 |       0.280218 |        0.269409 |      5981.02     |          26.2022  |
|       2 |     46.5421  |     45.247  |    2.97554 |   10.3718 |       0.279034 |        0.353509 |      4462.77     |          14.2062  |
|       3 |      2.45768 |     28.1763 |    2.35242 |   39.7941 |       0.287077 |        0.279006 |         3.3306   |           4.11062 |
|       4 |      5.31077 |     40.2966 |    2.08564 |   40.6234 |       0.290168 |        0.270955 |       323.234    |           3.23242 |
|       5 |      3.83112 |     47.1138 |    2.40274 |   28.7558 |       0.272059 |        0.286776 |       204.116    |           1.83316 |
|       6 |      1.7062  |     50.6368 |    1.66066 |   52.0739 |       0.265612 |        0.262688 |         4.2183   |           6.7498  |
|       7 |      2.32029 |     52.9175 |    1.61405 |   48.0907 |       0.269855 |        0.262187 |        74.5435   |           2.81956 |
|       8 |      1.71226 |     57.4341 |    1.52563 |   51.8348 |       0.269977 |        0.275169 |        31.9256   |           2.30331 |
|       9 |      1.24847 |     61.7085 |    1.71111 |   47.0422 |       0.260667 |        0.262739 |         2.16539  |           7.20298 |
|      10 |      1.17126 |     64.1386 |    1.54441 |   53.7456 |       0.258446 |        0.261422 |         2.12773  |           3.14339 |
|      11 |      1.12263 |     65.8604 |    1.57361 |   53.8954 |       0.258221 |        0.260711 |         2.15547  |           1.82167 |
|      12 |      1.06093 |     67.6797 |    1.63827 |   52.4343 |       0.255769 |        0.258818 |         1.32299  |           1.83124 |
|      13 |      1.04278 |     68.3439 |    1.67575 |   52.3339 |       0.25523  |        0.257745 |         1.38783  |           1.85163 |
|      14 |      1.02892 |     68.8349 |    1.68594 |   52.5703 |       0.25478  |        0.257494 |         1.39146  |           1.62087 |
|      15 |      1.00075 |     69.6739 |    1.68507 |   53.6065 |       0.253674 |        0.257089 |         0.937753 |           1.78814 |

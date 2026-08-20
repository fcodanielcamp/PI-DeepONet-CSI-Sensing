# 🔬 Reporte Científico de Experimento: 2026-08-20_01-41-38

## 1. Resumen de Ejecución
- **Duración Total:** 86.68 minutos (5200.8 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-01B
- **Test Loss:** 8.385753134308507
- **Test Accuracy_Percent:** 15.899146477624344

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
|       1 |     2.81041  |     38.6503 |    1.50855 |   52.4648 |       0.306739 |        0.273755 |        73.8824   |          2.79567  |
|       2 |     1.18979  |     63.6264 |    1.50316 |   55.5408 |       0.263114 |        0.264481 |         1.41409  |          1.20069  |
|       3 |     1.12064  |     66.3609 |    1.54937 |   53.1512 |       0.259293 |        0.263113 |         2.35293  |          1.22753  |
|       4 |     1.01378  |     69.4673 |    1.69811 |   49.7725 |       0.258533 |        0.26203  |         0.720391 |          0.946358 |
|       5 |     0.963443 |     71.2521 |    1.55466 |   55.9941 |       0.257414 |        0.260808 |         0.498231 |          0.789001 |
|       6 |     0.895177 |     73.744  |    1.70925 |   54.5874 |       0.255719 |        0.25858  |         0.373446 |          0.792363 |
|       7 |     0.870132 |     74.7205 |    1.6867  |   55.0434 |       0.255458 |        0.258631 |         0.363758 |          0.797231 |
|       8 |     0.850735 |     75.472  |    1.71533 |   55.0331 |       0.255297 |        0.258412 |         0.358655 |          0.763585 |
|       9 |     0.814471 |     76.8534 |    1.71333 |   55.9095 |       0.254738 |        0.257718 |         0.322314 |          0.776169 |
|      10 |     0.802843 |     77.3297 |    1.83826 |   55.1928 |       0.254685 |        0.257729 |         0.319584 |          0.805831 |
|      11 |     0.793834 |     77.6731 |    1.73084 |   56.6847 |       0.255067 |        0.258117 |         0.317229 |          0.739122 |
|      12 |     0.772994 |     78.5387 |    1.79967 |   56.364  |       0.2548   |        0.257872 |         0.299659 |          0.788778 |
|      13 |     0.767012 |     78.7832 |    1.85232 |   55.0816 |       0.25477  |        0.257852 |         0.297757 |          0.786635 |
|      14 |     0.761806 |     78.9673 |    1.84809 |   55.081  |       0.254747 |        0.257811 |         0.296116 |          0.774749 |
|      15 |     0.749808 |     79.4653 |    1.83474 |   55.5378 |       0.254615 |        0.257723 |         0.28712  |          0.776269 |

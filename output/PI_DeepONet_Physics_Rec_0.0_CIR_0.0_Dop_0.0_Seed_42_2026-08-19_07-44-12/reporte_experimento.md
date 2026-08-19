# 🔬 Reporte Científico de Experimento: 2026-08-19_07-44-12

## 1. Resumen de Ejecución
- **Duración Total:** 85.74 minutos (5144.1 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-01B
- **Test Loss:** 8.392104464336182
- **Test Accuracy_Percent:** 15.58392118903536

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
|       1 |     1.07551  |     58.8336 |    1.3212  |   52.2055 |              0 |               0 |      4.37098e+07 |       6.21897e+07 |
|       2 |     0.838871 |     66.1457 |    1.39762 |   50.4615 |              0 |               0 |      4.73526e+07 |       4.85228e+07 |
|       3 |     0.77467  |     68.4503 |    1.37121 |   52.9825 |              0 |               0 |      2.96303e+07 |       2.4803e+07  |
|       4 |     0.737566 |     69.8156 |    1.4705  |   52.4642 |              0 |               0 |      2.07174e+07 |       1.95225e+07 |
|       5 |     0.686376 |     71.727  |    1.43959 |   52.3009 |              0 |               0 |      1.96974e+07 |       1.99513e+07 |
|       6 |     0.671216 |     72.3217 |    1.50536 |   52.3144 |              0 |               0 |      1.90299e+07 |       1.86395e+07 |
|       7 |     0.659843 |     72.7684 |    1.51307 |   52.7975 |              0 |               0 |      1.65316e+07 |       1.45049e+07 |
|       8 |     0.632299 |     73.7943 |    1.53727 |   53.286  |              0 |               0 |      1.53982e+07 |       1.55251e+07 |
|       9 |     0.625128 |     74.0816 |    1.58743 |   51.1644 |              0 |               0 |      1.44563e+07 |       1.48592e+07 |
|      10 |     0.618559 |     74.3355 |    1.5362  |   53.8574 |              0 |               0 |      1.28249e+07 |       1.22969e+07 |
|      11 |     0.603442 |     74.9372 |    1.56858 |   52.918  |              0 |               0 |      1.23312e+07 |       1.24007e+07 |
|      12 |     0.598799 |     75.0774 |    1.57687 |   52.9884 |              0 |               0 |      1.19944e+07 |       1.26418e+07 |
|      13 |     0.595542 |     75.2171 |    1.58111 |   53.1267 |              0 |               0 |      1.17203e+07 |       1.18618e+07 |
|      14 |     0.586899 |     75.5938 |    1.63847 |   51.4562 |              0 |               0 |      1.14332e+07 |       1.25732e+07 |
|      15 |     0.584982 |     75.678  |    1.62838 |   51.7271 |              0 |               0 |      1.15488e+07 |       1.19214e+07 |

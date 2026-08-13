# 🔬 Reporte Científico de Experimento: 2026-08-12_12-08-45

## 1. Resumen de Ejecución
- **Duración Total:** 46.49 minutos (2789.3 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-06
- **Test Loss:** 7.943152054020917
- **Test Accuracy_Percent:** 10.359710652688205

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
| `NUM_WORKERS` | `4` |
| `EPOCHS` | `15` |
| `LEARNING_RATE` | `0.001` |
| `LAMBDA_PHYS` | `0.0` |
| `LAMBDA_TEMP` | `0.1` |
| `LAMBDA_ENERGY` | `0.0` |
| `GAMMA` | `0.0` |
| `MODEL` | `PI-DeepONet_CORAL_Ordinal_Energy` |
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
|   epoch |   train_loss |   train_acc |   val_loss |   val_acc |   tr_loss_temp |   val_loss_temp |   tr_loss_energy |   val_loss_energy |
|--------:|-------------:|------------:|-----------:|----------:|---------------:|----------------:|-----------------:|------------------:|
|       1 |     1.80861  |     50.0207 |    1.49902 |   46.6921 |       0.000451 |        0.000224 |      3.36873e+07 |       3.52668e+07 |
|       2 |     1.19172  |     58.2324 |    1.50864 |   47.6715 |       4.8e-05  |        2.9e-05  |      1.85724e+07 |       1.35268e+07 |
|       3 |     1.10974  |     60.5422 |    1.44033 |   49.497  |       5.5e-05  |        4e-05    |      9.09074e+06 |       7.34861e+06 |
|       4 |     1.0645   |     61.951  |    1.53151 |   46.445  |       2.9e-05  |        8e-06    |      6.06305e+06 |       5.87814e+06 |
|       5 |     1.03393  |     62.8721 |    1.49738 |   49.7813 |       1.9e-05  |        1.3e-05  |      4.90726e+06 |       5.00222e+06 |
|       6 |     1.01183  |     63.561  |    1.61368 |   45.0999 |       2.5e-05  |        2.1e-05  |      4.06179e+06 |       4.54969e+06 |
|       7 |     0.96112  |     65.1319 |    1.60867 |   48.0585 |       7e-06    |        8e-06    |      4.11509e+06 |       4.84501e+06 |
|       8 |     0.947797 |     65.5257 |    1.62312 |   45.5956 |       5e-06    |        7e-06    |      4.06231e+06 |       4.79966e+06 |
|       9 |     0.939531 |     65.8131 |    1.67586 |   46.0315 |       6e-06    |        6e-06    |      3.90084e+06 |       4.49746e+06 |
|      10 |     0.911908 |     66.6157 |    1.65153 |   46.2833 |       3e-06    |        2e-06    |      4.16875e+06 |       4.99032e+06 |
|      11 |     0.905935 |     66.7908 |    1.67992 |   44.9813 |       2e-06    |        2e-06    |      4.31486e+06 |       5.05138e+06 |
|      12 |     0.900644 |     66.9818 |    1.68632 |   48.5383 |       2e-06    |        2e-06    |      4.28162e+06 |       4.94989e+06 |
|      13 |     0.886057 |     67.4219 |    1.69071 |   47.0191 |       2e-06    |        2e-06    |      4.55703e+06 |       5.70931e+06 |
|      14 |     0.882498 |     67.5137 |    1.71218 |   46.0989 |       1e-06    |        1e-06    |      4.71568e+06 |       5.60928e+06 |
|      15 |     0.879304 |     67.6251 |    1.69919 |   45.7572 |       1e-06    |        1e-06    |      4.85355e+06 |       5.79268e+06 |

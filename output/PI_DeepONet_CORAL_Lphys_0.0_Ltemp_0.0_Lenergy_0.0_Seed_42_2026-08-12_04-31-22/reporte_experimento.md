# 🔬 Reporte Científico de Experimento: 2026-08-12_04-31-22

## 1. Resumen de Ejecución
- **Duración Total:** 46.60 minutos (2796.3 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-06
- **Test Loss:** 7.802167354996688
- **Test Accuracy_Percent:** 10.221079278142291

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
| `LAMBDA_TEMP` | `0.0` |
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
|       1 |     1.80574  |     50.0719 |    1.46886 |   47.5773 |       0.267483 |        0.170548 |      2.91856e+07 |       2.80014e+07 |
|       2 |     1.18673  |     58.4131 |    1.40868 |   50.0901 |       0.049466 |        0.00647  |      1.44163e+07 |       1.03093e+07 |
|       3 |     1.10065  |     60.8905 |    1.39988 |   51.315  |       0.017307 |        0.001652 |      6.9596e+06  |       5.60466e+06 |
|       4 |     1.05657  |     62.2024 |    1.44968 |   47.2693 |       0.003264 |        0.000561 |      4.61965e+06 |       5.35498e+06 |
|       5 |     1.02611  |     63.1903 |    1.45466 |   48.4342 |       0.001301 |        0.001104 |      3.56733e+06 |       4.08738e+06 |
|       6 |     1.00481  |     63.7841 |    1.51489 |   46.3773 |       0.001301 |        0.001209 |      3.04969e+06 |       3.07967e+06 |
|       7 |     0.954182 |     65.3573 |    1.51364 |   47.9844 |       0.00031  |        0.000168 |      2.90607e+06 |       3.3417e+06  |
|       8 |     0.94188  |     65.7223 |    1.54535 |   47.3351 |       0.000336 |        0.00032  |      2.69132e+06 |       3.16079e+06 |
|       9 |     0.932498 |     66.0142 |    1.5781  |   47.0568 |       0.000321 |        0.000428 |      2.57373e+06 |       2.79632e+06 |
|      10 |     0.905734 |     66.8223 |    1.61194 |   46.3676 |       0.000183 |        0.000207 |      2.6665e+06  |       3.12737e+06 |
|      11 |     0.899389 |     66.9851 |    1.6626  |   45.0423 |       0.000223 |        0.000261 |      2.69648e+06 |       3.16433e+06 |
|      12 |     0.894242 |     67.1726 |    1.62353 |   47.8222 |       0.000229 |        0.000187 |      2.71693e+06 |       3.04491e+06 |
|      13 |     0.879624 |     67.6345 |    1.62352 |   47.573  |       0.000154 |        0.000115 |      2.76483e+06 |       3.33333e+06 |
|      14 |     0.875721 |     67.7216 |    1.67573 |   46.7916 |       0.000134 |        0.000139 |      2.81216e+06 |       3.32096e+06 |
|      15 |     0.872781 |     67.8304 |    1.66768 |   46.2674 |       0.000157 |        0.000149 |      2.86303e+06 |       3.48098e+06 |

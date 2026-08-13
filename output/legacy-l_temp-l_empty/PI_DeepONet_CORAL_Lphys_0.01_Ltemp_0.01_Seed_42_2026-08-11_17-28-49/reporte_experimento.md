# 🔬 Reporte Científico de Experimento: 2026-08-11_17-28-49

## 1. Resumen de Ejecución
- **Duración Total:** 46.12 minutos (2767.1 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-06
- **Test Loss:** 6.942433748712712
- **Test Accuracy_Percent:** 10.910825185593056

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
| `LAMBDA_PHYS` | `0.01` |
| `LAMBDA_TEMP` | `0.01` |
| `MODEL` | `PI-DeepONet_CORAL_Ordinal` |
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
|   epoch |   train_loss |   train_acc |   val_loss |   val_acc |   tr_loss_temp |   val_loss_temp |
|--------:|-------------:|------------:|-----------:|----------:|---------------:|----------------:|
|       1 |     1.83369  |     49.773  |    1.46179 |   52.1971 |        5.5e-05 |         3e-05   |
|       2 |     1.20422  |     58.0515 |    1.42487 |   47.5717 |        2.9e-05 |         2.5e-05 |
|       3 |     1.12216  |     60.2908 |    1.36017 |   53.4488 |        3.2e-05 |         1.4e-05 |
|       4 |     1.07731  |     61.6686 |    1.35017 |   50.3733 |        1.2e-05 |         6e-06   |
|       5 |     1.04597  |     62.6301 |    1.40716 |   52.6301 |        9e-06   |         1.6e-05 |
|       6 |     1.02339  |     63.2728 |    1.48296 |   47.9333 |        1.2e-05 |         8e-06   |
|       7 |     1.00494  |     63.8494 |    1.4206  |   50.7448 |        2.3e-05 |         4.7e-05 |
|       8 |     0.964017 |     65.0246 |    1.51194 |   49.4707 |        3e-06   |         2e-06   |
|       9 |     0.954496 |     65.3231 |    1.57835 |   47.9201 |        2e-06   |         1e-06   |
|      10 |     0.947296 |     65.5595 |    1.54381 |   49.1064 |        3e-06   |         1e-06   |
|      11 |     0.924131 |     66.2115 |    1.64226 |   47.7294 |        0       |         0       |
|      12 |     0.918478 |     66.3882 |    1.5359  |   49.7321 |        0       |         0       |
|      13 |     0.915166 |     66.5507 |    1.55637 |   49.3866 |        0       |         0       |
|      14 |     0.901858 |     66.8868 |    1.62096 |   48.393  |        0       |         0       |
|      15 |     0.898743 |     66.9962 |    1.61026 |   47.9248 |        0       |         0       |

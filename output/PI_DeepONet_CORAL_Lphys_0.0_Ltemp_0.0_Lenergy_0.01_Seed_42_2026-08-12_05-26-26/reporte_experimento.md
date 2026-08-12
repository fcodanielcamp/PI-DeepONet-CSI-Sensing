# 🔬 Reporte Científico de Experimento: 2026-08-12_05-26-26

## 1. Resumen de Ejecución
- **Duración Total:** 46.43 minutos (2785.6 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-06
- **Test Loss:** 7.286672607654099
- **Test Accuracy_Percent:** 10.75099709824311

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
| `LAMBDA_ENERGY` | `0.01` |
| `GAMMA` | `1.0` |
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
|       1 |      9.07558 |     30.7866 |    2.12275 |   43.9824 |        1.9e-05 |         4e-06   |       614.026    |           2.93126 |
|       2 |      6.1113  |     45.0096 |    3.44573 |   12.9034 |        3.9e-05 |         4.6e-05 |       421.81     |           4.6704  |
|       3 |      2.36904 |     38.5343 |    1.89482 |   44.5994 |        1.2e-05 |         1.4e-05 |        10.5715   |           1.8447  |
|       4 |      4.14132 |     42.2473 |    1.8532  |   45.2574 |        1.5e-05 |         1.1e-05 |       212.198    |           2.68988 |
|       5 |      2.43448 |     47.7662 |    1.77425 |   44.9685 |        1.3e-05 |         1.2e-05 |        71.9722   |           3.47373 |
|       6 |      1.87752 |     50.4553 |    1.5797  |   49.6503 |        1.5e-05 |         1.4e-05 |        32.3029   |           1.85561 |
|       7 |      1.37266 |     54.4866 |    1.503   |   48.8009 |        1.5e-05 |         1.5e-05 |         2.72202  |           1.47621 |
|       8 |      1.40334 |     56.1551 |    1.46304 |   49.8753 |        1.3e-05 |         1.1e-05 |        12.0922   |           2.90543 |
|       9 |      1.22068 |     57.9127 |    1.49901 |   45.9375 |        1.9e-05 |         2e-05   |         1.50878  |           1.61427 |
|      10 |      1.17886 |     59.0475 |    1.50488 |   46.6478 |        2.5e-05 |         2.3e-05 |         1.50035  |           1.44012 |
|      11 |      1.14713 |     59.9642 |    1.52061 |   48.0956 |        2.7e-05 |         2.2e-05 |         1.52206  |           1.3101  |
|      12 |      1.0903  |     61.3784 |    1.44301 |   50.3226 |        2.9e-05 |         2.6e-05 |         0.967753 |           1.18766 |
|      13 |      1.07682 |     61.8809 |    1.55874 |   47.0005 |        2.8e-05 |         2.5e-05 |         1.07975  |           1.08498 |
|      14 |      1.06486 |     62.2232 |    1.55107 |   45.5831 |        3e-05   |         2.7e-05 |         1.00446  |           1.4034  |
|      15 |      1.05717 |     62.5188 |    1.56032 |   46.4218 |        2.8e-05 |         2.7e-05 |         1.15444  |           1.09708 |

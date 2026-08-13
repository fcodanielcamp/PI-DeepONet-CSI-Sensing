# 🔬 Reporte Científico de Experimento: 2026-08-12_03-34-51

## 1. Resumen de Ejecución
- **Duración Total:** 46.49 minutos (2789.1 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-06
- **Test Loss:** 6.2016641276964695
- **Test Accuracy_Percent:** 11.062491320311567

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
| `LAMBDA_TEMP` | `0.001` |
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
|       1 |      9.15499 |     30.3557 |    2.22621 |   42.8749 |        1.9e-05 |         3e-06   |       619.679    |          2.40626  |
|       2 |      5.29276 |     39.7313 |    2.62938 |   33.2301 |        1.7e-05 |         1.2e-05 |       313.227    |          2.74058  |
|       3 |      2.04961 |     43.522  |    1.76321 |   47.0406 |        6e-06   |         5e-06   |         4.68421  |          3.07099  |
|       4 |      2.93132 |     45.2903 |    1.7812  |   46.3423 |        9e-06   |         8e-06   |       108.005    |          2.82943  |
|       5 |      1.49091 |     51.8526 |    1.64062 |   46.4348 |        1.4e-05 |         1.3e-05 |         2.22113  |          1.998    |
|       6 |      1.9391  |     50.3783 |    1.54701 |   47.894  |        1.2e-05 |         1.3e-05 |        36.9468   |          3.27287  |
|       7 |      1.29158 |     56.0526 |    1.44947 |   50.3292 |        1.7e-05 |         1.6e-05 |         1.75442  |          1.47389  |
|       8 |      1.2809  |     57.4305 |    1.5114  |   47.3722 |        1.5e-05 |         1.4e-05 |         5.36909  |          3.48681  |
|       9 |      1.18586 |     58.782  |    1.58811 |   44.913  |        1.7e-05 |         1.9e-05 |         1.48746  |          1.58877  |
|      10 |      1.15143 |     59.7474 |    1.53664 |   47.1064 |        1.9e-05 |         1.6e-05 |         1.43691  |          1.48821  |
|      11 |      1.09607 |     61.2092 |    1.61678 |   45.7794 |        2e-05   |         1.5e-05 |         1.03115  |          1.23762  |
|      12 |      1.08111 |     61.6412 |    1.52779 |   48.888  |        2e-05   |         1.5e-05 |         1.04293  |          1.45044  |
|      13 |      1.0705  |     61.9706 |    1.59931 |   46.9051 |        1.9e-05 |         1.8e-05 |         1.05247  |          1.14251  |
|      14 |      1.04304 |     62.7003 |    1.61333 |   45.3489 |        2e-05   |         1.7e-05 |         0.853183 |          1.04435  |
|      15 |      1.0363  |     62.8663 |    1.62597 |   45.413  |        2e-05   |         1.7e-05 |         0.859654 |          0.957275 |

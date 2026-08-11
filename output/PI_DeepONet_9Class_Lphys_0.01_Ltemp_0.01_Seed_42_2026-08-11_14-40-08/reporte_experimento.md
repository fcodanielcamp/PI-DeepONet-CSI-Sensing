# 🔬 Reporte Científico de Experimento: 2026-08-11_14-40-08

## 1. Resumen de Ejecución
- **Duración Total:** 45.93 minutos (2755.7 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-06
- **Test Loss:** 10.957963670122698
- **Test Accuracy_Percent:** 10.362634337212901

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
| `MODEL` | `PI-DeepONet_9Class_TempLoss` |
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
|       1 |     1.35935  |     51.5955 |    1.60724 |   37.3735 |        5.9e-05 |         2.5e-05 |
|       2 |     1.14432  |     61.0573 |    1.82624 |   36.1329 |        2.7e-05 |         1.3e-05 |
|       3 |     1.08047  |     63.4551 |    2.1084  |   34.6417 |        1.5e-05 |         1.3e-05 |
|       4 |     1.03781  |     64.9962 |    1.93749 |   38.0939 |        1.2e-05 |         4.4e-05 |
|       5 |     0.975947 |     67.105  |    2.49954 |   33.1554 |        3e-06   |         3e-06   |
|       6 |     0.95379  |     67.8195 |    2.28744 |   35.1398 |        2e-06   |         2e-06   |
|       7 |     0.935843 |     68.2702 |    2.46246 |   33.4969 |        3e-06   |         2e-06   |
|       8 |     0.898889 |     69.4281 |    2.51556 |   35.763  |        1e-06   |         1e-06   |
|       9 |     0.887493 |     69.7821 |    2.78403 |   34.2391 |        1e-06   |         1e-06   |
|      10 |     0.877486 |     69.9968 |    2.5668  |   36.5541 |        1e-06   |         1e-06   |
|      11 |     0.855692 |     70.6584 |    2.83231 |   34.1878 |        1e-06   |         1e-06   |
|      12 |     0.84903  |     70.8064 |    2.67385 |   36.4181 |        1e-06   |         1e-06   |
|      13 |     0.843036 |     71.0047 |    3.12535 |   32.8758 |        1e-06   |         1e-06   |
|      14 |     0.831238 |     71.3098 |    3.05369 |   33.8663 |        2e-06   |         2e-06   |
|      15 |     0.827749 |     71.4146 |    3.00797 |   34.4459 |        2e-06   |         1e-06   |

# 🔬 Reporte Científico de Experimento: 2026-08-12_13-47-17

## 1. Resumen de Ejecución
- **Duración Total:** 46.73 minutos (2804.0 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-06
- **Test Loss:** 7.102190645402203
- **Test Accuracy_Percent:** 10.61053842086926

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
| `GAMMA` | `2.0` |
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
|       1 |      9.03484 |     31.6553 |    2.18566 |   44.7671 |        3.9e-05 |         3.3e-05 |        613.69    |          10.171   |
|       2 |      3.89584 |     39.3883 |    2.32866 |   39.6487 |        4e-05   |         2e-05   |        167.569   |          15.0311  |
|       3 |      1.8291  |     47.4315 |    1.79199 |   46.7215 |        2.6e-05 |         5.2e-05 |          8.58583 |           8.42345 |
|       4 |      2.62953 |     45.3239 |    1.67806 |   47.3037 |        2.8e-05 |         3.8e-05 |         76.9021  |           6.44219 |
|       5 |      1.89409 |     49.9206 |    1.64186 |   46.6077 |        3.9e-05 |         4.9e-05 |         30.26    |           6.41915 |
|       6 |      1.39292 |     54.8295 |    1.5469  |   48.4064 |        5.1e-05 |         5e-05   |          5.5981  |           5.14846 |
|       7 |      1.28223 |     57.303  |    1.46232 |   49.1702 |        4.4e-05 |         5e-05   |          5.06533 |           4.6516  |
|       8 |      1.21733 |     58.9555 |    1.51683 |   47.8907 |        4.6e-05 |         5.7e-05 |          4.67078 |           4.65199 |
|       9 |      1.17971 |     59.9626 |    1.56198 |   45.8748 |        5.2e-05 |         6e-05   |          4.45181 |           5.48841 |
|      10 |      1.15315 |     60.6811 |    1.54145 |   47.6787 |        5.2e-05 |         4.9e-05 |          4.321   |           4.49068 |
|      11 |      1.10148 |     61.885  |    1.65111 |   44.6156 |        5.3e-05 |         5.4e-05 |          3.45351 |           3.95419 |
|      12 |      1.08876 |     62.2383 |    1.55068 |   49.6828 |        5.2e-05 |         5e-05   |          3.45747 |           4.07493 |
|      13 |      1.07994 |     62.5273 |    1.62901 |   46.9839 |        5.1e-05 |         5.5e-05 |          3.42249 |           4.09599 |
|      14 |      1.0533  |     63.1753 |    1.62048 |   46.8616 |        5.1e-05 |         5.4e-05 |          3.03462 |           3.7321  |
|      15 |      1.04735 |     63.3206 |    1.6367  |   46.5735 |        5.1e-05 |         5.4e-05 |          3.02593 |           3.70259 |

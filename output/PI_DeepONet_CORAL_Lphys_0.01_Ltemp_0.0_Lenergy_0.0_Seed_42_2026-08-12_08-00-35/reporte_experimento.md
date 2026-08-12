# 🔬 Reporte Científico de Experimento: 2026-08-12_08-00-35

## 1. Resumen de Ejecución
- **Duración Total:** 46.47 minutos (2788.4 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-06
- **Test Loss:** 7.229665563741565
- **Test Accuracy_Percent:** 10.583372518827309

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
|       1 |     1.8328   |     49.8122 |    1.45469 |   51.5903 |        7.2e-05 |         2.4e-05 |         13205.7  |          11565.5  |
|       2 |     1.20378  |     58.1001 |    1.46447 |   46.2945 |        3.3e-05 |         2.1e-05 |         10938.2  |          17521.7  |
|       3 |     1.12286  |     60.3161 |    1.37948 |   51.9082 |        3.2e-05 |         1.2e-05 |         10045.7  |          10653.8  |
|       4 |     1.07798  |     61.6548 |    1.37737 |   49.3088 |        1.6e-05 |         5e-06   |          8981.72 |           8283.86 |
|       5 |     1.04651  |     62.6268 |    1.44453 |   49.1139 |        9e-06   |         2e-06   |          7915.06 |           9238.85 |
|       6 |     1.02407  |     63.2948 |    1.52067 |   47.4577 |        1.4e-05 |         3e-06   |          7338.09 |           7120.2  |
|       7 |     1.00624  |     63.8061 |    1.52427 |   48.275  |        2.6e-05 |         7e-06   |          7006.76 |           7242.7  |
|       8 |     0.965286 |     64.951  |    1.57385 |   47.9157 |        2e-06   |         2e-06   |          6640.33 |           7029.09 |
|       9 |     0.955767 |     65.2626 |    1.59827 |   47.6533 |        2e-06   |         1e-06   |          6577.68 |           6932.41 |
|      10 |     0.949027 |     65.4802 |    1.60431 |   48.1173 |        2e-06   |         1e-06   |          6594.42 |           6887.66 |
|      11 |     0.927097 |     66.1161 |    1.63188 |   47.741  |        0       |         0       |          6542.72 |           7055.78 |
|      12 |     0.921448 |     66.2757 |    1.59599 |   49.0298 |        0       |         0       |          6433.82 |           6947.32 |
|      13 |     0.918148 |     66.3932 |    1.61817 |   49.171  |        0       |         0       |          6102.89 |           6777.61 |
|      14 |     0.905176 |     66.7233 |    1.67318 |   47.7164 |        0       |         0       |          5563.62 |           5785.09 |
|      15 |     0.902158 |     66.8377 |    1.66211 |   46.7457 |        0       |         0       |          5348.62 |           6238.63 |

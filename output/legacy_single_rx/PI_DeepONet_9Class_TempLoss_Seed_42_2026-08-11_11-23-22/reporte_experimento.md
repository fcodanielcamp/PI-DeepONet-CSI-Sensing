# 🔬 Reporte Científico de Experimento: 2026-08-11_11-23-22

## 1. Resumen de Ejecución
- **Duración Total:** 14.07 minutos (844.4 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-06
- **Test Loss:** 19.437640171221112
- **Test Accuracy_Percent:** 10.003119164918367

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
| `MODEL` | `PI-DeepONet_9Class_TempLoss` |
| `DEVICE` | `cuda` |

---
## 4. Trazabilidad de Archivos de Datos (Hashes SHA-256)
| Identificador | Ruta | SHA-256 Hash |
| :--- | :--- | :--- |
| `metadata` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/dataset_mc1_rx1_80mhz_meta.npz` | `6faecd85b0b43ae7...` |
| `X_train` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_train_frames.dat` | `5da82590d4422945...` |
| `X_val` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_val_frames.dat` | `686b388a521ae147...` |
| `X_test` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_test_frames.dat` | `74471c351dde52bf...` |

---
## 5. Historial de Entrenamiento por Época
|   epoch |   train_loss |   train_acc |   val_loss |   val_acc |   tr_loss_temp |   val_loss_temp |
|--------:|-------------:|------------:|-----------:|----------:|---------------:|----------------:|
|       1 |     1.38369  |     49.8966 |    2.13462 |   32.1277 |       0.000234 |         7.1e-05 |
|       2 |     1.09851  |     63.0248 |    2.59061 |   24.3061 |       4.7e-05  |         3.8e-05 |
|       3 |     1.0267   |     65.9932 |    2.79713 |   25.444  |       4.7e-05  |         7.5e-05 |
|       4 |     0.980956 |     67.8592 |    3.52437 |   21.736  |       2.8e-05  |         2.5e-05 |
|       5 |     0.921226 |     70.0452 |    3.62632 |   24.696  |       1.2e-05  |         1.4e-05 |
|       6 |     0.90062  |     70.7242 |    3.35692 |   27.1233 |       1.1e-05  |         1.5e-05 |
|       7 |     0.881291 |     71.3252 |    4.07747 |   22.8002 |       1.5e-05  |         1.3e-05 |
|       8 |     0.844675 |     72.5648 |    5.46613 |   21.1084 |       9e-06    |         1e-05   |
|       9 |     0.831    |     72.985  |    4.82155 |   22.3093 |       7e-06    |         8e-06   |
|      10 |     0.818515 |     73.326  |    4.40024 |   23.7285 |       9e-06    |         8e-06   |
|      11 |     0.795047 |     74.0229 |    4.97605 |   22.8657 |       7e-06    |         7e-06   |
|      12 |     0.785847 |     74.3204 |    5.203   |   22.8664 |       7e-06    |         7e-06   |
|      13 |     0.778122 |     74.5828 |    5.20006 |   22.7859 |       6e-06    |         7e-06   |
|      14 |     0.762507 |     75.0415 |    5.35486 |   23.5058 |       7e-06    |         7e-06   |
|      15 |     0.756818 |     75.1877 |    5.30647 |   23.1259 |       6e-06    |         7e-06   |

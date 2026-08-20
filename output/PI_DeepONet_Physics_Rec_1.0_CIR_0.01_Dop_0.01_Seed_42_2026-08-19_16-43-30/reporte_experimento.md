# 🔬 Reporte Científico de Experimento: 2026-08-19_16-43-30

## 1. Resumen de Ejecución
- **Duración Total:** 86.77 minutos (5206.1 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-01B
- **Test Loss:** 7.299659659234443
- **Test Accuracy_Percent:** 15.694412527303664

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
| `LAMBDA_ENERGY` | `0.01` |
| `LAMBDA_MONO` | `0.001` |
| `LAMBDA_REC` | `1.0` |
| `LAMBDA_CIR` | `0.01` |
| `LAMBDA_DOP` | `0.01` |
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
|       1 |     2.35245  |     44.0668 |    1.54332 |   50.6605 |       0.294831 |        0.27047  |        46.7399   |          2.2105   |
|       2 |     1.16669  |     64.2461 |    1.64257 |   52.9034 |       0.262341 |        0.264107 |         1.30693  |          1.82035  |
|       3 |     1.05953  |     67.7484 |    1.64776 |   55.2695 |       0.25914  |        0.26416  |         0.763593 |          1.33847  |
|       4 |     0.997343 |     69.9646 |    1.60715 |   58.0116 |       0.257751 |        0.260301 |         0.587091 |          0.948877 |
|       5 |     0.922264 |     72.6573 |    1.64166 |   56.0958 |       0.255075 |        0.257842 |         0.416348 |          0.852507 |
|       6 |     0.89246  |     73.7531 |    1.64072 |   58.6251 |       0.254178 |        0.256698 |         0.388368 |          0.761396 |
|       7 |     0.86849  |     74.6707 |    1.64457 |   56.9182 |       0.253143 |        0.255136 |         0.37234  |          0.765482 |
|       8 |     0.825766 |     76.1208 |    1.70949 |   56.3071 |       0.248679 |        0.248858 |         0.330535 |          0.794114 |
|       9 |     0.804813 |     76.6114 |    1.78424 |   55.387  |       0.239444 |        0.23532  |         0.329817 |          0.784652 |
|      10 |     0.782584 |     76.9584 |    1.72592 |   56.4369 |       0.225737 |        0.221268 |         0.330771 |          0.775502 |
|      11 |     0.752186 |     77.8554 |    1.76907 |   55.2027 |       0.21737  |        0.217369 |         0.310134 |          0.78703  |
|      12 |     0.742903 |     78.0588 |    1.78546 |   55.4386 |       0.214314 |        0.214315 |         0.307894 |          0.774086 |
|      13 |     0.734879 |     78.2995 |    1.86112 |   54.5271 |       0.211795 |        0.212312 |         0.306372 |          0.851003 |
|      14 |     0.719933 |     78.8033 |    1.83414 |   55.3975 |       0.209694 |        0.210467 |         0.297723 |          0.790889 |
|      15 |     0.715537 |     78.953  |    1.84626 |   55.6384 |       0.208555 |        0.209455 |         0.297089 |          0.784274 |

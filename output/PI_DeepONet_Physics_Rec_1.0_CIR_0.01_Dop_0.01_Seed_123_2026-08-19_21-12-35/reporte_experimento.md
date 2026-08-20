# 🔬 Reporte Científico de Experimento: 2026-08-19_21-12-35

## 1. Resumen de Ejecución
- **Duración Total:** 86.81 minutos (5208.3 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-01B
- **Test Loss:** 5.167075650846276
- **Test Accuracy_Percent:** 15.771942163819658

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
| `SEED` | `123` |
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
|       1 |     2.41429  |     40.9389 |    1.48884 |   50.2024 |       0.289213 |        0.272645 |        43.1403   |          0.923528 |
|       2 |     1.23496  |     61.9073 |    1.56012 |   51.1952 |       0.262662 |        0.261947 |         1.48098  |          1.79774  |
|       3 |     1.09713  |     66.3899 |    1.67889 |   51.9743 |       0.257752 |        0.261446 |         0.953505 |          2.6379   |
|       4 |     1.01897  |     69.1942 |    1.71766 |   54.6812 |       0.256857 |        0.259923 |         0.733434 |          0.892371 |
|       5 |     0.940719 |     71.9686 |    1.83653 |   53.7578 |       0.255285 |        0.258172 |         0.5021   |          0.88657  |
|       6 |     0.912157 |     73.0653 |    1.8913  |   53.2299 |       0.255072 |        0.258367 |         0.484015 |          0.913722 |
|       7 |     0.889047 |     73.9545 |    1.79331 |   55.3942 |       0.254954 |        0.258047 |         0.459371 |          0.85116  |
|       8 |     0.848791 |     75.4682 |    1.87636 |   55.8139 |       0.254328 |        0.257413 |         0.364756 |          0.877737 |
|       9 |     0.835409 |     75.9877 |    1.91118 |   55.6259 |       0.254269 |        0.257342 |         0.359554 |          0.799064 |
|      10 |     0.824004 |     76.4321 |    1.84198 |   57.5992 |       0.254232 |        0.257194 |         0.349722 |          0.710086 |
|      11 |     0.800985 |     77.3464 |    2.01125 |   55.4831 |       0.254331 |        0.257434 |         0.302124 |          0.751638 |
|      12 |     0.793477 |     77.643  |    2.0606  |   55.788  |       0.254309 |        0.257433 |         0.297933 |          0.819428 |
|      13 |     0.786976 |     77.8981 |    2.13382 |   52.7738 |       0.254302 |        0.257373 |         0.292961 |          0.794344 |
|      14 |     0.773284 |     78.4441 |    2.09056 |   54.4741 |       0.254147 |        0.257222 |         0.272326 |          0.76078  |
|      15 |     0.769677 |     78.5792 |    2.04865 |   56.053  |       0.254138 |        0.257217 |         0.269636 |          0.74407  |

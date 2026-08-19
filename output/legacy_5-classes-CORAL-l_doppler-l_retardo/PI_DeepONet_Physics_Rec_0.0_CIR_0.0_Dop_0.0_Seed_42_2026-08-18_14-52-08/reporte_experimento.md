# 🔬 Reporte Científico de Experimento: 2026-08-18_14-52-08

## 1. Resumen de Ejecución
- **Duración Total:** 50.68 minutos (3040.6 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-01B
- **Test Loss:** 9.489017047778185
- **Test Accuracy_Percent:** 15.793065507900364

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
| `LAMBDA_ENERGY` | `0.0` |
| `LAMBDA_MONO` | `0.0` |
| `LAMBDA_REC` | `0.0` |
| `LAMBDA_CIR` | `0.0` |
| `LAMBDA_DOP` | `0.0` |
| `MODEL` | `PI-DeepONet_Physics_Fourier` |
| `DEVICE` | `cuda` |

---
## 4. Trazabilidad de Archivos de Datos (Hashes SHA-256)
| Identificador | Ruta | SHA-256 Hash |
| :--- | :--- | :--- |
| `metadata` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/dataset_mc1_rx1_80mhz_meta.npz` | `b4779fcd95fefadd...` |
| `X_train` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_train_frames.dat` | `17b8567418dad173...` |
| `X_val` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_val_frames.dat` | `a323d19d9f0be89a...` |
| `X_test` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_test_frames.dat` | `5a03477aa2b0d325...` |

---
## 5. Historial de Entrenamiento por Época
|   epoch |   train_loss |   train_acc |   val_loss |   val_acc |   tr_loss_phys |   val_loss_phys |   tr_loss_energy |   val_loss_energy |
|--------:|-------------:|------------:|-----------:|----------:|---------------:|----------------:|-----------------:|------------------:|
|       1 |     1.06746  |     59.0344 |    1.50391 |   45.6201 |              0 |               0 |      1.253e+07   |       1.19984e+07 |
|       2 |     0.829011 |     66.443  |    1.45584 |   50.5435 |              0 |               0 |      8.44793e+06 |       7.83095e+06 |
|       3 |     0.758756 |     68.9972 |    1.41206 |   52.5165 |              0 |               0 |      5.36744e+06 |       4.49259e+06 |
|       4 |     0.717865 |     70.4694 |    1.42777 |   54.1007 |              0 |               0 |      3.9565e+06  |       3.98128e+06 |
|       5 |     0.689461 |     71.566  |    1.42977 |   55.0679 |              0 |               0 |      3.01562e+06 |       3.02008e+06 |
|       6 |     0.669033 |     72.3538 |    1.44247 |   55.0677 |              0 |               0 |      2.48011e+06 |       2.46992e+06 |
|       7 |     0.627633 |     73.9435 |    1.6092  |   52.2027 |              0 |               0 |      2.39385e+06 |       2.5244e+06  |
|       8 |     0.616773 |     74.3267 |    1.67102 |   52.5505 |              0 |               0 |      2.3332e+06  |       2.50991e+06 |
|       9 |     0.608477 |     74.7074 |    1.49974 |   54.5716 |              0 |               0 |      2.36295e+06 |       2.55753e+06 |
|      10 |     0.586567 |     75.5526 |    1.62944 |   52.9321 |              0 |               0 |      2.45992e+06 |       2.67335e+06 |
|      11 |     0.581344 |     75.7117 |    1.54566 |   55.4066 |              0 |               0 |      2.53597e+06 |       2.79407e+06 |
|      12 |     0.576757 |     75.9366 |    1.59694 |   54.5222 |              0 |               0 |      2.52341e+06 |       2.9032e+06  |
|      13 |     0.565083 |     76.4015 |    1.62768 |   53.8236 |              0 |               0 |      2.63188e+06 |       3.0369e+06  |
|      14 |     0.561697 |     76.5177 |    1.65851 |   53.8227 |              0 |               0 |      2.72876e+06 |       2.94917e+06 |
|      15 |     0.559542 |     76.6199 |    1.66393 |   54.3001 |              0 |               0 |      2.75511e+06 |       2.99422e+06 |

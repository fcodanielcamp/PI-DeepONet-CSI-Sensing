# 🔬 Reporte Científico de Experimento: 2026-08-20_00-12-18

## 1. Resumen de Ejecución
- **Duración Total:** 85.69 minutos (5141.5 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-01B
- **Test Loss:** 8.686291312714006
- **Test Accuracy_Percent:** 15.672592809242275

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
| `SEED` | `777` |
| `NUM_CLASSES` | `5` |
| `BATCH_SIZE` | `256` |
| `EPOCHS` | `15` |
| `LEARNING_RATE` | `0.001` |
| `LAMBDA_ENERGY` | `0.0` |
| `LAMBDA_MONO` | `0.0` |
| `LAMBDA_REC` | `1.0` |
| `LAMBDA_CIR` | `0.0` |
| `LAMBDA_DOP` | `0.0` |
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
|       1 |     1.29288  |     60.3743 |    1.5639  |   51.0454 |       0.268854 |        0.263419 |      2.48578e+06 |       1.94008e+06 |
|       2 |     1.03151  |     68.3797 |    1.61803 |   51.3537 |       0.2593   |        0.262023 |      1.37325e+06 |       1.27071e+06 |
|       3 |     0.968589 |     70.7324 |    1.73191 |   52.2662 |       0.257953 |        0.260806 | 877937           |  949585           |
|       4 |     0.931506 |     72.1328 |    1.75074 |   51.7012 |       0.257293 |        0.260149 | 715656           |  707900           |
|       5 |     0.883545 |     73.9631 |    1.78062 |   54.7034 |       0.256219 |        0.259356 | 787345           |  947865           |
|       6 |     0.867224 |     74.558  |    1.89954 |   52.3112 |       0.25596  |        0.258902 | 823657           |  771133           |
|       7 |     0.854464 |     75.0513 |    1.86438 |   53.1876 |       0.255765 |        0.2587   | 793186           |       1.08292e+06 |
|       8 |     0.829136 |     76.0235 |    1.93756 |   52.2772 |       0.255218 |        0.258161 | 873024           |  985736           |
|       9 |     0.821239 |     76.327  |    1.94374 |   52.9151 |       0.255117 |        0.258144 | 934747           |  819228           |
|      10 |     0.814466 |     76.5687 |    2.01215 |   53.1506 |       0.255059 |        0.258061 | 933246           |       1.20463e+06 |
|      11 |     0.800616 |     77.145  |    2.05862 |   52.775  |       0.254801 |        0.257797 |      1.02339e+06 |  944570           |
|      12 |     0.796507 |     77.2905 |    2.03072 |   52.9011 |       0.254735 |        0.257729 |      1.01812e+06 |       1.20041e+06 |
|      13 |     0.793009 |     77.4293 |    2.09072 |   51.6704 |       0.254684 |        0.257737 |      1.06198e+06 |       1.098e+06   |
|      14 |     0.785251 |     77.7195 |    2.12842 |   51.9387 |       0.25454  |        0.257585 |      1.10509e+06 |       1.16947e+06 |
|      15 |     0.783022 |     77.8105 |    2.11175 |   52.2427 |       0.254505 |        0.257566 |      1.1289e+06  |       1.21357e+06 |

# 🔬 Reporte Científico de Experimento: 2026-08-19_18-13-58

## 1. Resumen de Ejecución
- **Duración Total:** 85.66 minutos (5139.3 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-01B
- **Test Loss:** 8.438497566920278
- **Test Accuracy_Percent:** 15.695805275265029

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
| `metadata` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/dataset_mc1_rx1_80mhz_meta.npz` | `532cc6f8e5410fa9...` |
| `X_train` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_train_frames.dat` | `e29dc479c9c23cbb...` |
| `X_val` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_val_frames.dat` | `0513eb4ed4a856a7...` |
| `X_test` | `/home/daniell/Desktop/PI-DeepONet/project/data/processed_tensors/X_test_frames.dat` | `e2b1497acf70184b...` |

---
## 5. Historial de Entrenamiento por Época
|   epoch |   train_loss |   train_acc |   val_loss |   val_acc |   tr_loss_phys |   val_loss_phys |   tr_loss_energy |   val_loss_energy |
|--------:|-------------:|------------:|-----------:|----------:|---------------:|----------------:|-----------------:|------------------:|
|       1 |     1.04003  |     59.7708 |    1.37325 |   49.0749 |              0 |               0 |      1.57686e+06 |       1.47694e+06 |
|       2 |     0.776147 |     68.2548 |    1.40217 |   52.6871 |              0 |               0 | 975479           |  848833           |
|       3 |     0.711026 |     70.7333 |    1.39883 |   53.4952 |              0 |               0 | 846456           |  741244           |
|       4 |     0.672613 |     72.1428 |    1.45908 |   52.777  |              0 |               0 | 737387           |  958120           |
|       5 |     0.624934 |     73.9595 |    1.54492 |   53.0597 |              0 |               0 | 744883           |  751299           |
|       6 |     0.609445 |     74.5698 |    1.56742 |   52.4703 |              0 |               0 | 759288           |  814152           |
|       7 |     0.596593 |     75.0817 |    1.58219 |   53.3427 |              0 |               0 | 726135           |  615549           |
|       8 |     0.572555 |     76.0338 |    1.66737 |   51.6781 |              0 |               0 | 762721           |  844286           |
|       9 |     0.56506  |     76.3072 |    1.64968 |   52.398  |              0 |               0 | 805110           |  853282           |
|      10 |     0.558575 |     76.5942 |    1.73037 |   51.0543 |              0 |               0 | 808522           |  852409           |
|      11 |     0.545095 |     77.1131 |    1.77766 |   51.6016 |              0 |               0 | 815145           |  929314           |
|      12 |     0.540883 |     77.2743 |    1.724   |   52.3928 |              0 |               0 | 872641           |  995596           |
|      13 |     0.537016 |     77.4515 |    1.76996 |   52.2681 |              0 |               0 | 848263           |  964654           |
|      14 |     0.529276 |     77.754  |    1.77011 |   51.8287 |              0 |               0 | 880793           |       1.032e+06   |
|      15 |     0.527557 |     77.7974 |    1.75779 |   52.0636 |              0 |               0 | 904418           |  931527           |

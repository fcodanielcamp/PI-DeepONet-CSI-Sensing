# 🔬 Reporte Científico de Experimento: 2026-08-12_01-18-43

## 1. Resumen de Ejecución
- **Duración Total:** 46.38 minutos (2782.9 segundos)
- **Épocas Ejecutadas:** 15
- **Test Target_Domain:** MC1-06
- **Test Loss:** 6.708458802401939
- **Test Accuracy_Percent:** 10.576794228646747

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
| `LAMBDA_ENERGY` | `0.001` |
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
|       1 |     3.05123  |     40.9934 |    1.79759 |   47.9824 |        1.5e-05 |         8e-06   |        622.6     |           6.14771 |
|       2 |     2.79513  |     43.3831 |    1.91985 |   41.9899 |        1.8e-05 |         4e-06   |        811.57    |           6.99905 |
|       3 |     1.52101  |     51.1384 |    1.54696 |   47.6558 |        8e-06   |         7e-06   |         12.2809  |           4.79166 |
|       4 |     1.52716  |     53.0771 |    1.48249 |   48.1908 |        1.1e-05 |         8e-06   |        106.447   |          11.7862  |
|       5 |     1.24959  |     56.9146 |    1.52236 |   48.6171 |        1e-05   |         7e-06   |          7.9669  |          21.2374  |
|       6 |     1.1845   |     58.7067 |    1.36404 |   51.4624 |        8e-06   |         5e-06   |          7.96939 |           5.38741 |
|       7 |     1.14257  |     59.9296 |    1.41374 |   50.8822 |        7e-06   |         8e-06   |          7.5933  |           9.75887 |
|       8 |     1.11147  |     60.9139 |    1.46679 |   49.402  |        8e-06   |         7e-06   |          7.87378 |           5.65443 |
|       9 |     1.09001  |     61.5462 |    1.65103 |   44.1364 |        9e-06   |         6e-06   |          7.66568 |           4.01965 |
|      10 |     1.04483  |     62.7533 |    1.55964 |   47.8836 |        1.3e-05 |         1.2e-05 |          3.71318 |           4.81504 |
|      11 |     1.03321  |     63.1246 |    1.61609 |   45.5703 |        1.2e-05 |         1.1e-05 |          3.75001 |           4.4013  |
|      12 |     1.02505  |     63.3746 |    1.47775 |   51.475  |        1.2e-05 |         9e-06   |          3.81415 |           7.32503 |
|      13 |     1.00194  |     64.0309 |    1.50748 |   49.4736 |        1.4e-05 |         1.1e-05 |          2.08853 |           2.40573 |
|      14 |     0.996476 |     64.1788 |    1.60125 |   46.238  |        1.4e-05 |         1.5e-05 |          2.09267 |           1.72722 |
|      15 |     0.991974 |     64.3017 |    1.61787 |   45.6589 |        1.4e-05 |         1.2e-05 |          2.11816 |           3.03491 |

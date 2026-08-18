from pathlib import Path
import numpy as np
import pandas as pd

# Definir ruta
BASE_DIR = Path(__file__).resolve().parent.parent
META_PATH = BASE_DIR / "data" / "processed_tensors" / "dataset_mc1_rx1_80mhz_meta.npz"

def inspect_label_distribution():
    if not META_PATH.exists():
        print(f"Error: No se encontró el archivo de metadatos en {META_PATH}")
        return

    meta = np.load(META_PATH, allow_pickle=True)

    tr_y = meta['train_y_people']
    va_y = meta['val_y_people']
    te_y = meta['test_y_people']

    print("=" * 70)
    print(" 📊 AUDITORÍA DE DISTRIBUCIÓN DE CLASES (N_People)")
    print("=" * 70)

    classes_tr = set(np.unique(tr_y))
    classes_va = set(np.unique(va_y))
    classes_te = set(np.unique(te_y))
    all_classes = sorted(list(classes_tr | classes_va | classes_te))

    data = []
    for c in all_classes:
        tr_cnt = np.sum(tr_y == c)
        va_cnt = np.sum(va_y == c)
        te_cnt = np.sum(te_y == c)
        
        data.append({
            'Clase (Personas)': c,
            'Train (Muestras)': tr_cnt,
            'Train (%)': round((tr_cnt / len(tr_y)) * 100, 2),
            'Val (Muestras)': va_cnt,
            'Val (%)': round((va_cnt / len(va_y)) * 100, 2),
            'Test (Muestras)': te_cnt,
            'Test (%)': round((te_cnt / len(te_y)) * 100, 2)
        })

    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print("=" * 70)

    # Verificación de inconsistencias
    missing_in_train = classes_te - classes_tr
    if missing_in_train:
        print(f"\n⚠️  ALERTA CRÍTICA: Hay clases en Test no presentes en Train: {missing_in_train}")
    else:
        print("\n✅ OK: Todas las clases presentes en Test fueron vistas durante Train.")

    print(f"\nResumen de Muestras:")
    print(f"  - Total Train: {len(tr_y):,} ventanas")
    print(f"  - Total Val:   {len(va_y):,} ventanas")
    print(f"  - Total Test:  {len(te_y):,} ventanas")

if __name__ == "__main__":
    inspect_label_distribution()

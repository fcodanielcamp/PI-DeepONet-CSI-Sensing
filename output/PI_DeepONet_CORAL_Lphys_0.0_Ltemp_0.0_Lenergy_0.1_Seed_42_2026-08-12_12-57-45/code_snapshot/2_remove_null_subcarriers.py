import os
from pathlib import Path
import pandas as pd
import scipy.io as scipy_io
import numpy as np

# ==============================================================================
# CONFIGURACIÓN DE RUTAS DINÁMICAS (COMPATIBLE CON LINUX Y WINDOWS)
# ==============================================================================
SRC_DIR = Path(__file__).resolve().parent          # .../project/src
PROJECT_DIR = SRC_DIR.parent                        # .../project

DATA_RAW_DIR = PROJECT_DIR / "data" / "raw"
DATA_PREPROCESSED_DIR = PROJECT_DIR / "data" / "preprocessed" / "cleaned_sc"

SUMMARY_EXCEL_PATH = DATA_RAW_DIR / "Summary_Filtrado.xlsx"

def get_null_carrier_indices(bw: int, environment: str, total_sc: int) -> list:
    carrier_remove20_default = [0, 1, 2, 3, 32, 61, 62, 63]
    carrier_remove80A = [0, 1, 2, 3, 4, 5, 127, 128, 129, 130, 251, 252, 253, 254, 255]
    carrier_remove80B = [
        0, 1, 2, 3, 4, 5, 32, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
        96, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 160,
        187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 224, 251, 252, 253, 254, 255
    ]

    if total_sc == 56:
        carrier_remove20 = []
    else:
        carrier_remove20 = carrier_remove20_default

    if bw == 20:
        return carrier_remove20
    elif bw == 80:
        if environment == "Industrial Laboratory":
            return carrier_remove80B
        else:
            return carrier_remove80A
    return []

def remove_null_subcarriers():
    print("=== MÓDULO 2: ELIMINACIÓN DE SUBPORTADORAS NULAS (OPTIMIZADO EN ESPACIO) ===")

    if not SUMMARY_EXCEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo de resumen: {SUMMARY_EXCEL_PATH}.")

    df = pd.read_excel(SUMMARY_EXCEL_PATH)
    DATA_PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    skipped_count = 0

    for idx, row in df.iterrows():
        file_name = row['File']
        src_file = DATA_RAW_DIR / file_name

        if not src_file.exists():
            skipped_count += 1
            continue

        mat_data = scipy_io.loadmat(src_file)

        csi_key = None
        for k in mat_data.keys():
            if k.upper() == 'CSI':
                csi_key = k
                break

        if csi_key is None:
            skipped_count += 1
            continue

        csi_matrix = mat_data[csi_key]

        if csi_matrix.size == 0 or len(csi_matrix.shape) < 2:
            skipped_count += 1
            continue

        bw = int(row['BW'])
        env = str(row['Enviroment'])
        sc_count = csi_matrix.shape[-1]

        indices_to_remove = get_null_carrier_indices(bw, env, sc_count)

        if len(indices_to_remove) > 0:
            cleaned_csi = np.delete(csi_matrix, indices_to_remove, axis=-1)
        else:
            cleaned_csi = csi_matrix

        # Si es complejo, asegurar precisión simple (complex64) para ahorrar espacio
        if np.iscomplexobj(cleaned_csi):
            cleaned_csi = cleaned_csi.astype(np.complex64)

        mat_data[csi_key] = cleaned_csi

        dst_file = DATA_PREPROCESSED_DIR / file_name

        # ACTIVA COMPRESIÓN AL GUARDAR
        scipy_io.savemat(dst_file, mat_data, do_compression=True)

        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Procesados {processed_count}/{len(df)} archivos...")

    print(f"\n¡Módulo 2 Completado!")
    print(f"- Archivos guardados comprimidos en: '{DATA_PREPROCESSED_DIR}' ({processed_count} archivos)")

    summary_dst = DATA_PREPROCESSED_DIR / "Summary_Filtrado.xlsx"
    df.to_excel(summary_dst, index=False)

if __name__ == "__main__":
    remove_null_subcarriers()

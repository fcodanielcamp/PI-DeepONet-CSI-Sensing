import os
import shutil
from pathlib import Path
import pandas as pd

# ==============================================================================
# CONFIGURACIÓN DE RUTAS DINÁMICAS (COMPATIBLE CON LINUX Y WINDOWS)
# ==============================================================================
SRC_DIR = Path(__file__).resolve().parent          # .../project/src
PROJECT_DIR = SRC_DIR.parent                        # .../project
ROOT_DIR = PROJECT_DIR.parent                       # .../PI-DeepONet

# 1. Ubicación de metadata y descarga original
META_DIR = ROOT_DIR / "EHUNAM"
FULL_DOWNLOAD_DIR = Path.home() / "baseEHUNAM"

# 2. Ubicación de destino para datos filtrados (raw)
DATA_RAW_DIR = PROJECT_DIR / "data" / "raw"

EXCEL_SOURCE_PATH = META_DIR / "Summary.xlsx"

def filter_and_copy_dataset():
    print("=== MÓDULO 1: FILTRADO Y COPIADO DE ARCHIVOS ÚTILES ===")
    print(f"Leyendo metadata desde: {EXCEL_SOURCE_PATH}")

    if not EXCEL_SOURCE_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo de resumen en: {EXCEL_SOURCE_PATH}")

    df = pd.read_excel(EXCEL_SOURCE_PATH)

    # Filtrado: Campaña MC1, 80 MHz, Aplicaciones 'E' (vacío) y 'PC' (conteo de personas)
    filtered_df = df[
        (df['Set'].astype(str).str.startswith('MC1')) &
        (df['BW'] == 80) &
        (df['Application'].isin(['E', 'PC']))
    ].copy()

    print(f"Archivos identificados tras el filtro: {len(filtered_df)} de {len(df)} totales.")

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Escaneando archivos .mat en: {FULL_DOWNLOAD_DIR} ...")
    file_map = {}
    for root, _, files in os.walk(FULL_DOWNLOAD_DIR):
        for file in files:
            if file.endswith('.mat'):
                file_map[file] = Path(root) / file

    copied_count = 0
    missing_count = 0

    for idx, row in filtered_df.iterrows():
        file_name = row['File']
        if file_name in file_map:
            src_path = file_map[file_name]
            dst_path = DATA_RAW_DIR / file_name
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        else:
            missing_count += 1

        if copied_count % 500 == 0 and copied_count > 0:
            print(f"Copiados {copied_count}/{len(filtered_df)} archivos...")

    print(f"\n¡Módulo 1 Completado!")
    print(f"- Archivos copiados intactos a: '{DATA_RAW_DIR}' ({copied_count} archivos)")
    if missing_count > 0:
        print(f"- Advertencia: {missing_count} archivos no se encontraron en la descarga original.")

    excel_out = DATA_RAW_DIR / "Summary_Filtrado.xlsx"
    filtered_df.to_excel(excel_out, index=False)
    print(f"- Resumen filtrado guardado en: '{excel_out}'")

if __name__ == "__main__":
    filter_and_copy_dataset()

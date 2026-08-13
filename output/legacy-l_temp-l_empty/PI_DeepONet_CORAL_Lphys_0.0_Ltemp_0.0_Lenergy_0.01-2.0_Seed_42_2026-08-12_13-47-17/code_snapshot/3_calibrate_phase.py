import os
from pathlib import Path
import pandas as pd
import scipy.io as scipy_io
import numpy as np
from scipy.signal import savgol_filter

# ==============================================================================
# CONFIGURACIÓN DE RUTAS DINÁMICAS (COMPATIBLE CON LINUX Y WINDOWS)
# ==============================================================================
SRC_DIR = Path(__file__).resolve().parent          # .../project/src
PROJECT_DIR = SRC_DIR.parent                        # .../project
DATA_DIR = PROJECT_DIR / "data"

INPUT_CLEANED_SC_DIR = DATA_DIR / "preprocessed" / "cleaned_sc"
OUTPUT_PHASE_CALIBRATED_DIR = DATA_DIR / "preprocessed" / "phase_calibrated"

SUMMARY_EXCEL_PATH = INPUT_CLEANED_SC_DIR / "Summary_Filtrado.xlsx"

def tsfr_phase_calibration(csi_matrix: np.ndarray) -> np.ndarray:
    """
    Aplica la técnica Two-Stage Phase Cleaning (TSFR) sobre una matriz CSI.

    Etapas:
      1. Regresión lineal por trama para eliminar SFO, STO y CFO.
      2. Filtro Savitzky-Golay a lo largo del tiempo para atenuar ruido.
      3. Corrección de saltos (Phase Unwrapping) entre subportadoras.

    Parameters:
        csi_matrix: Matriz compleja de forma (N_CSI, N_subcarriers)

    Returns:
        np.ndarray: Matriz CSI reconstruida con fase calibrada y amplitud intacta.
    """
    amplitude = np.abs(csi_matrix)
    raw_phase = np.angle(csi_matrix)

    n_csi, n_sc = raw_phase.shape

    # --------------------------------------------------------------------------
    # ETAPA 1: Regresión Lineal por Trama (Eliminación de SFO, STO y CFO)
    # --------------------------------------------------------------------------
    # Generación robusta de índices centrados (ej. -120 a +120 para 241 subportadoras)
    subcarrier_indices = np.arange(n_sc) - (n_sc // 2)

    m_mean = np.mean(subcarrier_indices)
    m_dev = subcarrier_indices - m_mean
    m_var = np.sum(m_dev ** 2)

    # Promedios y variaciones de fase por cada trama recibida
    phi_mean = np.mean(raw_phase, axis=1, keepdims=True)
    phi_dev = raw_phase - phi_mean

    # Estimación de pendiente (a) y offset (b) por mínimos cuadrados
    a = np.sum(phi_dev * m_dev[None, :], axis=1, keepdims=True) / m_var
    b = phi_mean - a * m_mean

    # Sustracción del sesgo lineal
    lin_calibrated_phase = raw_phase - (a * subcarrier_indices[None, :] + b)

    # --------------------------------------------------------------------------
    # ETAPA 2: Filtrado temporal Savitzky-Golay (Atenuación de ruido)
    # --------------------------------------------------------------------------
    # Se aplica el filtro a lo largo del eje del tiempo (axis=0)
    window_length = 15 if n_csi >= 15 else (n_csi - 1 if n_csi % 2 == 0 else n_csi)
    if window_length > 3:
        sg_phase = savgol_filter(lin_calibrated_phase, window_length=window_length, polyorder=2, axis=0)
    else:
        sg_phase = lin_calibrated_phase

    # --------------------------------------------------------------------------
    # ETAPA 3: Unwrapping / Corrección de saltos entre subportadoras
    # --------------------------------------------------------------------------
    calibrated_phase = np.unwrap(sg_phase, axis=1)

    # Reconstrucción de la matriz CSI compleja: H = |H| * exp(j * phase_calibrated)
    calibrated_csi = amplitude * np.exp(1j * calibrated_phase)

    return calibrated_csi.astype(np.complex64)


def calibrate_phase_dataset():
    print("=== MÓDULO 3: CALIBRACIÓN DE FASE CSI (MÉTODOS TSFR) ===")

    if not SUMMARY_EXCEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo de resumen: {SUMMARY_EXCEL_PATH}.")

    df = pd.read_excel(SUMMARY_EXCEL_PATH)
    OUTPUT_PHASE_CALIBRATED_DIR.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    skipped_count = 0

    for idx, row in df.iterrows():
        file_name = row['File']
        src_file = INPUT_CLEANED_SC_DIR / file_name

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

        # Aplicar saneamiento de fase TSFR
        calibrated_csi = tsfr_phase_calibration(csi_matrix)

        # Guardar matriz saneada en el diccionario MATLAB
        mat_data[csi_key] = calibrated_csi

        dst_file = OUTPUT_PHASE_CALIBRATED_DIR / file_name
        scipy_io.savemat(dst_file, mat_data, do_compression=True)

        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Calibrados {processed_count}/{len(df)} archivos...")

    print(f"\n¡Módulo 3 Completado!")
    print(f"- Archivos con fase calibrada guardados en: '{OUTPUT_PHASE_CALIBRATED_DIR}' ({processed_count} archivos)")

    summary_dst = OUTPUT_PHASE_CALIBRATED_DIR / "Summary_Filtrado.xlsx"
    df.to_excel(summary_dst, index=False)
    print(f"- Resumen copiado a: '{summary_dst}'")

if __name__ == "__main__":
    calibrate_phase_dataset()

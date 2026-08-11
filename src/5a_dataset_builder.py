import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.io as scipy_io
from sklearn.model_selection import train_test_split

# ==============================================================================
# CONFIGURACIÓN DE RUTAS DINÁMICAS (PORTABLE SERVIDOR)
# ==============================================================================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
INPUT_CALIBRATED_DIR = DATA_DIR / "preprocessed" / "phase_calibrated"
OUTPUT_TENSORS_DIR = DATA_DIR / "processed_tensors"
SUMMARY_EXCEL = INPUT_CALIBRATED_DIR / "Summary_Filtrado.xlsx"

# Parámetros base
WINDOW_SIZE = 25
STRIDE = 12
TARGET_BW = 80
TARGET_CAMPAIGN = "MC1"
EXPECTED_SUBCARRIERS = 241
MIN_CSI_FRAMES = 400

# Dominio reservado explícitamente para evaluación Cross-Domain
TEST_SET_DOMAINS = ['MC1_06', 'MC1-06']

def get_scalar_safe(mat_dict: dict, key: str, default=0):
    if key in mat_dict:
        val = mat_dict[key]
        if hasattr(val, 'size') and val.size > 0:
            return val.flat[0]
    return default

def get_str_safe(mat_dict: dict, key: str, default=""):
    if key in mat_dict:
        val = mat_dict[key]
        if hasattr(val, 'size') and val.size > 0:
            return str(val.flat[0])
    return default

def process_file_group_memmap(file_list, input_dir, split_name, output_dir):
    valid_files = []
    window_starts = []
    labels_people = []
    labels_empty = []
    file_ids = []
    set_ids = []
    env_ids = []
    date_ids = []
    global_frame_offset = 0

    # 1. Pasada de inspección y conteo de tramas válidas (Acepta TODOS los receptores Rx)
    for file_name in file_list:
        mat_path = input_dir / file_name
        if not mat_path.exists():
            continue
        mat_data = scipy_io.loadmat(mat_path)
        bw_val = get_scalar_safe(mat_data, 'BW', 80)
        app_val = get_str_safe(mat_data, 'Application', 'E')
        set_val = get_str_safe(mat_data, 'Set', "")
        env_val = get_str_safe(mat_data, 'Environment', "")
        date_val = get_str_safe(mat_data, 'Date', "")

        # 🚀 Aceptamos cualquier Rx siempre que cumpla con el ancho de banda y la campaña
        if bw_val != TARGET_BW:
            continue
        if app_val not in ['E', 'PC'] or not set_val.startswith(TARGET_CAMPAIGN):
            continue

        csi_key = [k for k in mat_data.keys() if k.upper() == 'CSI'][0]
        csi_matrix = mat_data[csi_key]
        n_frames, n_sc = csi_matrix.shape

        if n_sc != EXPECTED_SUBCARRIERS or n_frames < MIN_CSI_FRAMES or n_frames < WINDOW_SIZE:
            continue

        n_people = get_scalar_safe(mat_data, 'N_People', 0) if app_val == 'PC' else 0
        is_empty = 1 if app_val == 'E' else 0

        local_starts = np.arange(0, n_frames - WINDOW_SIZE + 1, STRIDE)
        global_starts = local_starts + global_frame_offset
        n_win = len(global_starts)

        window_starts.extend(global_starts)
        labels_people.extend([int(n_people)] * n_win)
        labels_empty.extend([is_empty] * n_win)
        file_ids.extend([file_name] * n_win)
        set_ids.extend([set_val] * n_win)
        env_ids.extend([env_val] * n_win)
        date_ids.extend([date_val] * n_win)

        valid_files.append((file_name, n_frames))
        global_frame_offset += n_frames

    total_frames = global_frame_offset
    if len(valid_files) == 0:
        return None, None, None, None, None, None, None, None, None, 0

    # 2. Crear archivo mapeado en disco (float16)
    dat_path = output_dir / f"X_{split_name}_frames.dat"
    X_memmap = np.memmap(
        dat_path,
        dtype='float16',
        mode='w+',
        shape=(total_frames, EXPECTED_SUBCARRIERS, 2)
    )

    # 3. Segunda pasada: Escribir datos directamente en disco
    current_offset = 0
    for file_name, n_frames in valid_files:
        mat_path = input_dir / file_name
        mat_data = scipy_io.loadmat(mat_path)
        csi_key = [k for k in mat_data.keys() if k.upper() == 'CSI'][0]
        csi_matrix = mat_data[csi_key]

        amp = np.abs(csi_matrix).astype(np.float16)
        phase = np.angle(csi_matrix).astype(np.float16)

        X_memmap[current_offset: current_offset + n_frames, :, 0] = amp
        X_memmap[current_offset: current_offset + n_frames, :, 1] = phase

        current_offset += n_frames

    X_memmap.flush()

    return (
        dat_path,
        (total_frames, EXPECTED_SUBCARRIERS, 2),
        np.array(window_starts, dtype=np.int64),
        np.array(labels_people, dtype=np.int64),
        np.array(labels_empty, dtype=np.int64),
        np.array(file_ids, dtype=object),
        np.array(set_ids, dtype=object),
        np.array(env_ids, dtype=object),
        np.array(date_ids, dtype=object),
        len(valid_files)
    )

def build_dataset():
    print("=== MÓDULO 5A: DATASET MULTI-RX CROSS-DOMAIN (TODAS LAS ANTENAS) ===")
    if not SUMMARY_EXCEL.exists():
        raise FileNotFoundError(f"No se encontró el resumen de metadatos en {SUMMARY_EXCEL}.")

    df_summary = pd.read_excel(SUMMARY_EXCEL)
    OUTPUT_TENSORS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 🚀 Incluir archivos de TODOS los receptores Rx
    df_filtered = df_summary[
        (df_summary['BW'] == TARGET_BW) &
        (df_summary['Application'].isin(['E', 'PC'])) &
        (df_summary['Set'].astype(str).str.startswith(TARGET_CAMPAIGN))
    ].copy()

    # 2. Separación de Dominio Objetivo (Test: MC1-06) vs Dominio Origen (Train/Val)
    test_domains_normalized = [s.replace('_', '-') for s in TEST_SET_DOMAINS]
    set_series_normalized = df_filtered['Set'].astype(str).str.replace('_', '-')
    test_mask = set_series_normalized.isin(test_domains_normalized)

    test_df = df_filtered[test_mask].copy()
    source_df = df_filtered[~test_mask].copy()

    # Asignar etiqueta de personas para la estratificación
    people_col = 'N_People' if 'N_People' in source_df.columns else ('N_people' if 'N_people' in source_df.columns else None)
    if people_col:
        source_df['file_label'] = source_df.apply(
            lambda r: 0 if str(r['Application']).strip() == 'E' else int(r[people_col]), axis=1
        )
    else:
        source_df['file_label'] = source_df['Application'].apply(
            lambda x: 0 if str(x).strip() == 'E' else 1
        )

    # Garantizar alineación indexada exacta entre archivos y etiquetas
    file_df = source_df.groupby('File', as_index=False)['file_label'].first()

    # División ESTRATIFICADA por número de personas (80% Train / 20% Val)
    train_files, val_files = train_test_split(
        file_df['File'].values,
        test_size=0.20,
        random_state=42,
        stratify=file_df['file_label'].values
    )

    test_files = test_df['File'].unique()

    print(f"\nPartición Multi-Rx Cross-Domain Asignada:")
    print(f"  Dominios de Train: {source_df['Set'].unique()} ({len(train_files)} archivos)")
    print(f"  Dominios de Val: {source_df['Set'].unique()} ({len(val_files)} archivos)")
    print(f"  Dominios de Test (Target No Visto): {test_df['Set'].unique()} ({len(test_files)} archivos)")

    # 3. Procesar aisladamente cada grupo en disco
    print("\nProcesando conjunto TRAIN (Multi-Rx)...")
    tr_path, tr_shape, tr_starts, tr_people, tr_empty, \
    tr_ids, tr_sets, tr_envs, tr_dates, n_tr_files = \
        process_file_group_memmap(train_files, INPUT_CALIBRATED_DIR, "train", OUTPUT_TENSORS_DIR)

    print("Procesando conjunto VAL (Multi-Rx)...")
    va_path, va_shape, va_starts, va_people, va_empty, \
    va_ids, va_sets, va_envs, va_dates, n_va_files = \
        process_file_group_memmap(val_files, INPUT_CALIBRATED_DIR, "val", OUTPUT_TENSORS_DIR)

    print("Procesando conjunto TEST (Multi-Rx Target No Visto)...")
    te_path, te_shape, te_starts, te_people, te_empty, \
    te_ids, te_sets, te_envs, te_dates, n_te_files = \
        process_file_group_memmap(test_files, INPUT_CALIBRATED_DIR, "test", OUTPUT_TENSORS_DIR)

    # 4. Normalización Z-score calculada EXCLUSIVAMENTE sobre TRAIN
    X_tr_mem = np.memmap(tr_path, dtype='float16', mode='r', shape=tr_shape)
    sample_indices = np.random.choice(tr_shape[0], size=min(100000, tr_shape[0]), replace=False)
    sample_data = X_tr_mem[sample_indices].astype(np.float32)

    mean_amp = float(np.mean(sample_data[:, :, 0]))
    std_amp = float(np.std(sample_data[:, :, 0])) + 1e-8
    mean_phase = float(np.mean(sample_data[:, :, 1]))
    std_phase = float(np.std(sample_data[:, :, 1])) + 1e-8

    print(f"\nEstadísticas Z-Score obtenidas estrictamente del conjunto TRAIN:")
    print(f"  Amplitud: Mean = {mean_amp:.4f}, Std = {std_amp:.4f}")
    print(f"  Fase: Mean = {mean_phase:.4f}, Std = {std_phase:.4f}")

    # 5. Normalizar archivos en disco por bloques
    print("\nAplicando estandarización Z-score en disco por bloques...")
    chunk_size = 500000
    for i in range(0, tr_shape[0], chunk_size):
        X_tr_mem_rw = np.memmap(tr_path, dtype='float16', mode='r+', shape=tr_shape)
        chunk = X_tr_mem_rw[i: i + chunk_size, :, :].astype(np.float32)
        chunk[:, :, 0] = (chunk[:, :, 0] - mean_amp) / std_amp
        chunk[:, :, 1] = (chunk[:, :, 1] - mean_phase) / std_phase
        X_tr_mem_rw[i: i + chunk_size, :, :] = chunk.astype(np.float16)
        X_tr_mem_rw.flush()

    for path_i, shape_i in [(va_path, va_shape), (te_path, te_shape)]:
        X_mem_rw = np.memmap(path_i, dtype='float16', mode='r+', shape=shape_i)
        for i in range(0, shape_i[0], chunk_size):
            chunk = X_mem_rw[i: i + chunk_size, :, :].astype(np.float32)
            chunk[:, :, 0] = (chunk[:, :, 0] - mean_amp) / std_amp
            chunk[:, :, 1] = (chunk[:, :, 1] - mean_phase) / std_phase
            X_mem_rw[i: i + chunk_size, :, :] = chunk.astype(np.float16)
            X_mem_rw.flush()

    # 6. Guardar metadatos e índices
    save_meta_path = OUTPUT_TENSORS_DIR / "dataset_mc1_rx1_80mhz_meta.npz"
    np.savez_compressed(
        save_meta_path,
        tr_shape=tr_shape, train_starts=tr_starts,
        train_y_people=tr_people, train_y_empty=tr_empty,
        train_file_ids=tr_ids, train_set_ids=tr_sets,
        train_env_ids=tr_envs, train_date_ids=tr_dates,
        va_shape=va_shape, val_starts=va_starts,
        val_y_people=va_people, val_y_empty=va_empty,
        val_file_ids=va_ids, val_set_ids=va_sets,
        val_env_ids=va_envs, val_date_ids=va_dates,
        te_shape=te_shape, test_starts=te_starts,
        test_y_people=te_people, test_y_empty=te_empty,
        test_file_ids=te_ids, test_set_ids=te_sets,
        test_env_ids=te_envs, test_date_ids=te_dates,
        stats=np.array([mean_amp, std_amp, mean_phase, std_phase], dtype=np.float32)
    )

    print(f"\nResumen de Ventanas Multi-Rx Generadas:")
    print(f"  - Train: {len(tr_starts)} ventanas ({n_tr_files} archivos)")
    print(f"  - Val: {len(va_starts)} ventanas ({n_va_files} archivos)")
    print(f"  - Test: {len(te_starts)} ventanas ({n_te_files} archivos Dominio Target No Visto)")
    print(f"\n¡Dataset Multi-Rx reconstruido exitosamente en: '{save_meta_path}'!")

if __name__ == "__main__":
    build_dataset()

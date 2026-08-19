import os
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.io as scipy_io
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# ==============================================================================
# CONFIGURACIÓN DE RUTAS DINÁMICAS (COMPATIBLE CON LINUX Y WINDOWS)
# ==============================================================================
SRC_DIR = Path(__file__).resolve().parent          # .../project/src
PROJECT_DIR = SRC_DIR.parent                        # .../project

DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
TSFR_DIR = DATA_DIR / "preprocessed" / "phase_calibrated"

OUTPUT_DIR = PROJECT_DIR / "output" / "validation"
SUMMARY_EXCEL = TSFR_DIR / "Summary_Filtrado.xlsx"

# ------------------------------------------------------------------------------
# FUNCIONES AUXILIARES DE EXTRACCIÓN SEGURA
# ------------------------------------------------------------------------------
def get_scalar_safe(mat_dict: dict, key: str, default=0):
    """Extrae de forma segura un número escalar desde una variable de MATLAB."""
    if key in mat_dict:
        val = mat_dict[key]
        if hasattr(val, 'size') and val.size > 0:
            return val.flat[0]
    return default

def get_str_safe(mat_dict: dict, key: str, default=""):
    """Extrae de forma segura una cadena de texto desde una variable de MATLAB."""
    if key in mat_dict:
        val = mat_dict[key]
        if hasattr(val, 'size') and val.size > 0:
            return str(val.flat[0])
    return default

# ------------------------------------------------------------------------------
# 1. CARGA Y PARSING DE ARCHIVOS
# ------------------------------------------------------------------------------
def load_mat_file(filepath: Path) -> dict:
    """Lee el archivo .mat y extrae el diccionario de variables de EHUNAM de forma robusta."""
    mat_data = scipy_io.loadmat(filepath)

    csi_key = [k for k in mat_data.keys() if k.upper() == 'CSI'][0]

    parsed = {
        'CSI': mat_data[csi_key],
        'Timestamp': mat_data.get('Timestamp', None),
        'Rx': get_scalar_safe(mat_data, 'Rx', default=1),
        'BW': get_scalar_safe(mat_data, 'BW', default=80),
        'Set': get_str_safe(mat_data, 'Set', default=''),
        'Application': get_str_safe(mat_data, 'Application', default=''),
        'N_People': get_scalar_safe(mat_data, 'N_People', default=0)
    }
    return parsed

# ------------------------------------------------------------------------------
# 2. CONTROL DE CALIDAD BÁSICO
# ------------------------------------------------------------------------------
def check_min_csi(csi_matrix: np.ndarray, min_frames: int = 400) -> bool:
    """Verifica si el archivo tiene el número mínimo aceptable de tramas CSI (400)."""
    n_frames = csi_matrix.shape[0]
    return n_frames >= min_frames

# ------------------------------------------------------------------------------
# 3. SEPARACIÓN DE AMPLITUD Y FASE
# ------------------------------------------------------------------------------
def split_amp_phase(csi_matrix: np.ndarray):
    """Separa la matriz compleja CSI en magnitud y fase (radianes)."""
    amp = np.abs(csi_matrix)
    phase = np.angle(csi_matrix)
    return amp, phase

# ------------------------------------------------------------------------------
# 4. MEDIDA DE CONTINUIDAD DE FASE (SUAVIDAD ENTRE SUBPORTADORAS)
# ------------------------------------------------------------------------------
def phase_diff_stats(phase: np.ndarray):
    """
    Calcula las diferencias absolutas de fase entre subportadoras adyacentes: |phi_{s+1} - phi_s|.
    Retorna la media, desviación estándar y la proporción de saltos/outliers (> 1.0 rad).
    """
    diffs = np.abs(np.diff(phase, axis=1))
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs))
    outlier_ratio = float(np.mean(diffs > 1.0))
    return mean_diff, std_diff, outlier_ratio

# ------------------------------------------------------------------------------
# 5. MEDIDA DE ESTABILIDAD EN VACÍO (CLASE E)
# ------------------------------------------------------------------------------
def empty_room_stability(phase: np.ndarray) -> float:
    """
    Calcula la varianza temporal promedio de la fase en la clase E (cuarto vacío).
    Un valor menor indica una señal más estable y libre de ruido residual/CFO.
    """
    temporal_var_per_sc = np.var(phase, axis=0)
    residual_noise_score = float(np.mean(temporal_var_per_sc))
    return residual_noise_score

# ------------------------------------------------------------------------------
# 6. EVALUACIÓN DE SEPARABILIDAD (E vs PC)
# ------------------------------------------------------------------------------
def pc_separability_test(features: np.ndarray, labels: np.ndarray) -> float:
    """
    Evalúa la separabilidad entre 'E' (vacío) y 'PC' (conteo de personas)
    usando un clasificador simple de bajo orden (Regresión Logística con validación cruzada).
    Retorna el AUC score.
    """
    if len(np.unique(labels)) < 2:
        return 0.5

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    for train_idx, val_idx in skf.split(features, labels):
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        clf = LogisticRegression(max_iter=500)
        clf.fit(X_train, y_train)

        preds = clf.predict_proba(X_val)[:, 1]
        auc_scores.append(roc_auc_score(y_val, preds))

    return float(np.mean(auc_scores))

# ------------------------------------------------------------------------------
# 7. GENERACIÓN DE REPORTES Y GRÁFICOS
# ------------------------------------------------------------------------------
def build_report(df_results: pd.DataFrame, auc_raw: float, auc_tsfr: float, output_dir: Path):
    """Exporta el reporte en CSV y genera figuras visuales comparativas."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Exportar CSV
    csv_path = output_dir / "validation_summary.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nReporte detallado por archivo exportado a: '{csv_path}'")

    # 1. Gráfico de Estabilidad de Fase en Clase E
    df_e = df_results[df_results['Application'] == 'E']
    if not df_e.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(['Crudo (Raw)', 'Preprocesado (TSFR)'],
                [df_e['Raw_Noise_Score'].mean(), df_e['TSFR_Noise_Score'].mean()],
                color=['#d9534f', '#5cb85c'])
        plt.ylabel('Ruido Residual de Fase (Varianza Temporal)')
        plt.title('Estabilidad Temporal de Fase en Clase E (Habitación Vacía)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / "empty_room_stability.png", dpi=300)
        plt.close()

    # 2. Gráfico de Continuidad de Fase (Outliers entre Subportadoras)
    plt.figure(figsize=(8, 5))
    plt.bar(['Crudo (Raw)', 'Preprocesado (TSFR)'],
            [df_results['Raw_Outlier_Ratio'].mean() * 100, df_results['TSFR_Outlier_Ratio'].mean() * 100],
            color=['#d9534f', '#0275d8'])
    plt.ylabel('Proporción de Discontinuidades (%)')
    plt.title('Discontinuidades de Fase entre Subportadoras (|Δφ| > 1.0 rad)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "phase_continuity.png", dpi=300)
    plt.close()

    # 3. Gráfico de Separabilidad E vs PC (AUC Score)
    plt.figure(figsize=(7, 5))
    plt.bar(['Crudo (Raw)', 'Preprocesado (TSFR)'], [auc_raw, auc_tsfr], color=['#f0ad4e', '#5cb85c'])
    plt.ylim(0.5, 1.0)
    plt.ylabel('AUC Score (E vs PC)')
    plt.title('Capacidad de Separabilidad entre Vacío (E) y Personas (PC)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "separability_auc.png", dpi=300)
    plt.close()

# ------------------------------------------------------------------------------
# EJECUCIÓN DEL PIPELINE DE VALIDACIÓN
# ------------------------------------------------------------------------------
def run_validation_pipeline():
    print("=== PIPELINE AUTOMÁTICO DE VALIDACIÓN DE PREPROCESADO (CONSENSUS) ===")

    if not SUMMARY_EXCEL.exists():
        raise FileNotFoundError(f"No se encontró {SUMMARY_EXCEL}. Ejecuta los módulos anteriores.")

    df_summary = pd.read_excel(SUMMARY_EXCEL)

    records = []

    features_raw = []
    features_tsfr = []
    labels = []

    valid_files_count = 0
    anomalous_files_count = 0

    for idx, row in df_summary.iterrows():
        file_name = row['File']
        raw_path = RAW_DIR / file_name
        tsfr_path = TSFR_DIR / file_name

        if not (raw_path.exists() and tsfr_path.exists()):
            continue

        data_raw = load_mat_file(raw_path)
        data_tsfr = load_mat_file(tsfr_path)

        # Check de número mínimo de tramas CSI (400)
        is_valid_raw = check_min_csi(data_raw['CSI'], min_frames=400)
        is_valid_tsfr = check_min_csi(data_tsfr['CSI'], min_frames=400)

        if not (is_valid_raw and is_valid_tsfr):
            anomalous_files_count += 1
            print(f"Alerta: Archivo {file_name} contiene menos de 400 tramas CSI. Se omite.")
            continue

        valid_files_count += 1

        amp_raw, phase_raw = split_amp_phase(data_raw['CSI'])
        amp_tsfr, phase_tsfr = split_amp_phase(data_tsfr['CSI'])

        # Métricas de continuidad de fase
        mean_d_raw, std_d_raw, out_r_raw = phase_diff_stats(phase_raw)
        mean_d_tsfr, std_d_tsfr, out_r_tsfr = phase_diff_stats(phase_tsfr)

        # Métricas de estabilidad en cuarto vacío (Clase E)
        noise_score_raw = empty_room_stability(phase_raw) if row['Application'] == 'E' else np.nan
        noise_score_tsfr = empty_room_stability(phase_tsfr) if row['Application'] == 'E' else np.nan

        records.append({
            'File': file_name,
            'Application': row['Application'],
            'Set': row['Set'],
            'Rx': data_tsfr['Rx'],
            'N_Frames': data_tsfr['CSI'].shape[0],
            'Raw_Mean_Phase_Diff': mean_d_raw,
            'TSFR_Mean_Phase_Diff': mean_d_tsfr,
            'Raw_Outlier_Ratio': out_r_raw,
            'TSFR_Outlier_Ratio': out_r_tsfr,
            'Raw_Noise_Score': noise_score_raw,
            'TSFR_Noise_Score': noise_score_tsfr
        })

        # Extracción de características simples para separabilidad E vs PC
        feat_r = [np.mean(amp_raw), np.std(amp_raw), np.mean(phase_raw), np.std(phase_raw)]
        feat_t = [np.mean(amp_tsfr), np.std(amp_tsfr), np.mean(phase_tsfr), np.std(phase_tsfr)]

        features_raw.append(feat_r)
        features_tsfr.append(feat_t)
        labels.append(1 if row['Application'] == 'PC' else 0)

        if valid_files_count % 100 == 0:
            print(f"Evaluados {valid_files_count}/{len(df_summary)} archivos...")

    df_results = pd.DataFrame(records)

    # Calcular test de separabilidad E vs PC
    auc_raw = pc_separability_test(np.array(features_raw), np.array(labels))
    auc_tsfr = pc_separability_test(np.array(features_tsfr), np.array(labels))

    # Imprimir Resumen en Consola
    print("\n" + "="*70)
    print("                    RESUMEN DE VALIDACIÓN TSFR")
    print("="*70)
    print(f"Total archivos evaluados válidos: {valid_files_count} (Anómalos descartados: {anomalous_files_count})")

    df_e_res = df_results[df_results['Application'] == 'E']
    if not df_e_res.empty:
        raw_noise = df_e_res['Raw_Noise_Score'].mean()
        tsfr_noise = df_e_res['TSFR_Noise_Score'].mean()
        noise_reduction = ((raw_noise - tsfr_noise) / raw_noise) * 100
        print(f"\n1. ESTABILIDAD EN CUARTO VACÍO (CLASE E):")
        print(f"   - Ruido Residual Fase Cruda (Varianza): {raw_noise:.4f}")
        print(f"   - Ruido Residual Fase TSFR  (Varianza): {tsfr_noise:.4f}")
        print(f"   - Reducción de ruido residual:         {noise_reduction:+.2f}%")

    raw_out = df_results['Raw_Outlier_Ratio'].mean() * 100
    tsfr_out = df_results['TSFR_Outlier_Ratio'].mean() * 100
    print(f"\n2. CONTINUIDAD DE FASE ENTRE SUBPORTADORAS:")
    print(f"   - Discontinuidades (|Δφ| > 1.0 rad) Crudas: {raw_out:.2f}%")
    print(f"   - Discontinuidades (|Δφ| > 1.0 rad) TSFR:   {tsfr_out:.2f}%")

    print(f"\n3. SEPARABILIDAD DE TAREA (CLASES E vs PC - AUC SCORE):")
    print(f"   - AUC Basal con Fase Cruda: {auc_raw:.4f}")
    print(f"   - AUC Basal con Fase TSFR:  {auc_tsfr:.4f}")
    print("="*70)

    # Generar reportes impresos y gráficos
    build_report(df_results, auc_raw, auc_tsfr, OUTPUT_DIR)
    print(f"¡Gráficos de validación guardados exitosamente en: '{OUTPUT_DIR}'!")

if __name__ == "__main__":
    run_validation_pipeline()

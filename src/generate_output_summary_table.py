import json
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"


def parse_diagnostics_report(report_path: Path) -> dict:
    """Parsea classification_report.txt para extraer Macro F1 y Balanced Acc de VAL y TEST."""
    diag_metrics = {
        "Val_Macro_F1": "N/A",
        "Val_Balanced_Acc": "N/A",
        "Test_Macro_F1": "N/A",
        "Test_Balanced_Acc": "N/A",
    }
    if not report_path.exists():
        return diag_metrics

    try:
        content = report_path.read_text(encoding="utf-8")
        current_split = None
        for line in content.splitlines():
            line_str = line.strip()
            if "SPLIT: VAL" in line_str:
                current_split = "Val"
            elif "SPLIT: TEST" in line_str:
                current_split = "Test"
            elif "SPLIT: TRAIN" in line_str:
                current_split = "Train"

            if current_split in ["Val", "Test"]:
                if "macro_f1 =" in line_str:
                    val = line_str.split("=")[1].strip()
                    diag_metrics[f"{current_split}_Macro_F1"] = (
                        f"{float(val):.4f}"
                    )
                elif "balanced_acc =" in line_str:
                    val = line_str.split("=")[1].strip()
                    diag_metrics[f"{current_split}_Balanced_Acc"] = (
                        f"{float(val):.4f}"
                    )
    except Exception as e:
        print(f"⚠️ Error al parsear diagnósticos en {report_path.parent}: {e}")

    return diag_metrics


def generate_output_summary_table():
    print(
        "=== RECOPILANDO TODAS LAS MÉTRICAS DE EXPERIMENTOS (OUTPUT DEPTH=1) ==="
    )

    if not OUTPUT_DIR.exists():
        print(f"⚠️ No se encontró el directorio {OUTPUT_DIR}")
        return

    records = []

    for folder in sorted(OUTPUT_DIR.iterdir()):
        if not folder.is_dir():
            continue

        folder_name = folder.name
        config_path = folder / "config.json"
        summary_path = folder / "summary.csv"
        metrics_history_path = folder / "metrics_history.csv"
        diag_report_path = folder / "diagnostics" / "classification_report.txt"

        # Hiperparámetros
        model_name = "N/A"
        seed = "N/A"
        lambda_phys = "N/A"
        lambda_temp = "N/A"
        lambda_energy = "N/A"
        gamma = "N/A"

        # Métricas Globales
        best_val_loss = "N/A"
        best_val_acc = "N/A"
        test_domain = "N/A"
        test_loss = "N/A"
        test_acc = "N/A"
        duration_min = "N/A"
        epochs = "N/A"

        # Métricas Físicas de la Mejor Época
        val_loss_temp = "N/A"
        val_loss_energy = "N/A"

        # 1. Leer Config
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)

                hp = config_data.get("Hyperparameters", {})
                model_name = hp.get("MODEL", "N/A")
                seed = hp.get("SEED", "N/A")
                lambda_phys = hp.get("LAMBDA_PHYS", "N/A")
                lambda_temp = hp.get("LAMBDA_TEMP", "N/A")
                lambda_energy = hp.get("LAMBDA_ENERGY", "N/A")
                gamma = hp.get("GAMMA", "N/A")
            except Exception as e:
                print(f"⚠️ Error leyendo config.json en {folder_name}: {e}")

        # 2. Leer Summary
        if summary_path.exists():
            try:
                df_sum = pd.read_csv(summary_path)
                if not df_sum.empty:
                    row = df_sum.iloc[0]

                    if "Best_Val_Loss" in df_sum.columns and pd.notna(
                        row["Best_Val_Loss"]
                    ):
                        best_val_loss = f"{float(row['Best_Val_Loss']):.4f}"
                    if "Best_Val_Acc" in df_sum.columns and pd.notna(
                        row["Best_Val_Acc"]
                    ):
                        best_val_acc = f"{float(row['Best_Val_Acc']):.2f}%"
                    if "Test_Target_Domain" in df_sum.columns and pd.notna(
                        row["Test_Target_Domain"]
                    ):
                        test_domain = str(row["Test_Target_Domain"])
                    if "Test_Loss" in df_sum.columns and pd.notna(
                        row["Test_Loss"]
                    ):
                        test_loss = f"{float(row['Test_Loss']):.4f}"
                    if "Test_Accuracy_Percent" in df_sum.columns and pd.notna(
                        row["Test_Accuracy_Percent"]
                    ):
                        test_acc = f"{float(row['Test_Accuracy_Percent']):.2f}%"
                    if "Duracion_Minutos" in df_sum.columns and pd.notna(
                        row["Duracion_Minutos"]
                    ):
                        duration_min = f"{float(row['Duracion_Minutos']):.2f}"
                    if "Epocas_Completadas" in df_sum.columns and pd.notna(
                        row["Epocas_Completadas"]
                    ):
                        epochs = int(row["Epocas_Completadas"])
            except Exception as e:
                print(f"⚠️ Error leyendo summary.csv en {folder_name}: {e}")

        # 3. Leer Métricas Físicas por Época (Mejor Época de Validación)
        if metrics_history_path.exists():
            try:
                df_mh = pd.read_csv(metrics_history_path)
                if not df_mh.empty and "val_loss" in df_mh.columns:
                    best_idx = df_mh["val_loss"].idxmin()
                    best_row = df_mh.loc[best_idx]

                    if "val_loss_temp" in df_mh.columns and pd.notna(
                        best_row["val_loss_temp"]
                    ):
                        val_loss_temp = (
                            f"{float(best_row['val_loss_temp']):.6f}"
                        )
                    if "val_loss_energy" in df_mh.columns and pd.notna(
                        best_row["val_loss_energy"]
                    ):
                        val_loss_energy = (
                            f"{float(best_row['val_loss_energy']):.6f}"
                        )
            except Exception as e:
                print(
                    f"⚠️ Error leyendo metrics_history.csv en {folder_name}: {e}"
                )

        # 4. Leer Diagnósticos Post-Hoc (Macro F1 & Balanced Acc)
        diag_metrics = parse_diagnostics_report(diag_report_path)

        # 5. Acumular Registro Completo
        if config_path.exists() or summary_path.exists():
            records.append({
                "Experimento": folder_name,
                "Modelo": model_name,
                "L_phys": lambda_phys,
                "L_temp": lambda_temp,
                "L_energy": lambda_energy,
                "Gamma": gamma,
                "Seed": seed,
                "Best_Val_Loss": best_val_loss,
                "Best_Val_Acc": best_val_acc,
                "Val_Loss_Temp": val_loss_temp,
                "Val_Loss_Energy": val_loss_energy,
                "Val_Macro_F1": diag_metrics["Val_Macro_F1"],
                "Val_Balanced_Acc": diag_metrics["Val_Balanced_Acc"],
                "Test_Domain": test_domain,
                "Test_Loss": test_loss,
                "Test_Acc": test_acc,
                "Test_Macro_F1": diag_metrics["Test_Macro_F1"],
                "Test_Balanced_Acc": diag_metrics["Test_Balanced_Acc"],
                "Duracion_Min": duration_min,
                "Epocas": epochs,
            })

    if not records:
        print("⚠️ No se encontraron carpetas válidas con métricas en output/.")
        return

    # Exportar resultados
    df_out = pd.DataFrame(records)

    csv_out = OUTPUT_DIR / "tabla_resumen_experimentos.csv"
    md_out = OUTPUT_DIR / "tabla_resumen_experimentos.md"

    df_out.to_csv(csv_out, index=False)

    md_content = "# 📊 Tabla Consolidada Completa de Experimentos\n\n"
    md_content += "Incluye pérdidas globales, términos físicos por época y métricas de diagnóstico (Macro F1 y Balanced Acc).\n\n"
    md_content += df_out.to_markdown(index=False)
    md_content += "\n"

    with open(md_out, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n✅ Proceso completado exitosamente con TODAS las métricas:")
    print(f"  👉 CSV actualizado en: {csv_out}")
    print(f"  👉 Markdown actualizado en: {md_out}\n")


if __name__ == "__main__":
    generate_output_summary_table()

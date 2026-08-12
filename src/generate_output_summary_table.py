import json
from pathlib import Path
import pandas as pd

# CONFIGURACIÓN DE RUTAS DINÁMICAS (PORTABLE EN SERVIDOR)
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"


def generate_output_summary_table():
    print("=== RECOPILANDO TABLA CONSOLIDADA DE EXPERIMENTOS (OUTPUT DEPTH=1) ===")

    if not OUTPUT_DIR.exists():
        print(f"⚠️ No se encontró el directorio {OUTPUT_DIR}")
        return

    records = []

    # 1. Exploración con estricta PROFUNDIDAD 1 (solo subcarpetas directas de output/)
    for folder in sorted(OUTPUT_DIR.iterdir()):
        if not folder.is_dir():
            continue

        folder_name = folder.name
        config_path = folder / "config.json"
        summary_path = folder / "summary.csv"

        # Valores por defecto en caso de carpetas incompletas o legacy
        model_name = "N/A"
        seed = "N/A"
        lambda_phys = "N/A"
        lambda_temp = "N/A"
        lambda_energy = "N/A"
        gamma = "N/A"

        best_val_loss = "N/A"
        best_val_acc = "N/A"
        test_loss = "N/A"
        test_acc = "N/A"
        duration_min = "N/A"
        epochs = "N/A"

        # 2. Extraer Hiperparámetros desde config.json
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

        # 3. Extraer Métricas Finales desde summary.csv
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

        # Acumular solo carpetas que sean experimentos válidos
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
                "Test_Loss": test_loss,
                "Test_Acc": test_acc,
                "Duracion_Min": duration_min,
                "Epocas": epochs,
            })

    if not records:
        print("⚠️ No se encontraron carpetas válidas con métricas en output/.")
        return

    # 4. Construir DataFrame y Guardar
    df_out = pd.DataFrame(records)

    csv_out = OUTPUT_DIR / "tabla_resumen_experimentos.csv"
    md_out = OUTPUT_DIR / "tabla_resumen_experimentos.md"

    # Exportar a CSV
    df_out.to_csv(csv_out, index=False)

    # Exportar a Markdown
    md_content = "# 📊 Tabla Consolidada de Experimentos\n\n"
    md_content += "Extraída automáticamente a Profundidad 1 sobre la carpeta `output/`.\n\n"
    md_content += df_out.to_markdown(index=False)
    md_content += "\n"

    with open(md_out, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n✅ Proceso completado exitosamente:")
    print(f"  👉 CSV listo en: {csv_out}")
    print(f"  👉 Markdown listo en: {md_out}\n")

    # Mostrar vista previa tabular en la terminal
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    generate_output_summary_table()

import json
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_DIR / "runs"

def generate_global_report():
    print("=== GENERANDO REPORTE GLOBAL DEFINITIVO ===")
    
    records = []
    
    if not RUNS_DIR.exists():
        print(f"⚠️ No se encontró el directorio {RUNS_DIR}")
        return

    for run_folder in sorted(RUNS_DIR.iterdir()):
        if run_folder.is_dir():
            folder_name = run_folder.name
            config_path = run_folder / "config.json"
            summary_path = run_folder / "summary.csv"
            diag_dir = run_folder / "diagnostics"
            
            # 1. Inferir modelo
            model_name = "Unknown"
            if "PI_DEEPONET" in folder_name.upper() or "PIDEEPONET" in folder_name.upper():
                model_name = "PI-DeepONet"
            elif "CNN" in folder_name.upper():
                model_name = "Pure CNN"
                
            # 2. Inferir semilla
            seed = "N/A"
            lower_name = folder_name.lower()
            if "seed_" in lower_name:
                try:
                    parts = lower_name.split("seed_")
                    seed = parts[1].split("_")[0]
                except:
                    pass

            # 3. Refinar con config.json si existe
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    hp = config.get("hyperparameters", {})
                    if "MODEL" in hp:
                        model_name = hp["MODEL"]
                    if "SEED" in hp:
                        seed = hp["SEED"]
                except Exception:
                    pass

            # 4. Leer summary.csv para extraer pérdidas y precisiones reales
            val_loss, val_acc = "N/A", "N/A"
            if summary_path.exists():
                try:
                    df_sum = pd.read_csv(summary_path)
                    if not df_sum.empty:
                        last_row = df_sum.iloc[-1]
                        # Normalizar nombres de columnas a minúsculas para búsqueda segura
                        df_sum.columns = [c.lower() for c in df_sum.columns]
                        
                        for col in ["val_loss", "v_loss", "loss"]:
                            if col in df_sum.columns:
                                val_loss = f"{float(last_row[col]):.4f}"
                                break
                                
                        for col in ["val_acc", "v_acc", "accuracy", "accuracy_percent"]:
                            if col in df_sum.columns:
                                val_acc = f"{float(last_row[col]):.2f}%"
                                break
                except Exception as e:
                    print(f"Error leyendo summary en {folder_name}: {e}")

            # 5. Buscar imagen de matriz de confusión con ruta relativa correcta para VS Code
            # Como el md estará en runs/, la ruta desde runs/ es: folder_name/diagnostics/imagen.png
            cm_image_md = "_No disponible_"
            if diag_dir.exists():
                images = list(diag_dir.glob("confusion_matrix_*.png"))
                if images:
                    img_name = images[0].name
                    # Ruta relativa limpia desde el archivo global_comparison_report.md (que vive en runs/)
                    cm_image_md = f"![Matriz de Confusión]({folder_name}/diagnostics/{img_name})"

            records.append({
                "Experiment": folder_name,
                "Model": model_name,
                "Seed": seed,
                "Val_Loss": val_loss,
                "Val_Acc": val_acc,
                "CM": cm_image_md
            })

    if not records:
        print("⚠️ No se encontraron experimentos.")
        return

    df = pd.DataFrame(records)

    # Construir Markdown
    md_content = "# 📊 Reporte Global de Experimentos: Comparativa y Diagnósticos\n\n"
    md_content += "Este reporte consolida el rendimiento de los modelos y sus respectivas matrices de confusión por corrida.\n\n"
    
    md_content += "## 📋 Tabla Comparativa General\n\n"
    md_content += df[["Experiment", "Model", "Seed", "Val_Loss", "Val_Acc"]].to_markdown(index=False)
    md_content += "\n\n## 📈 Matrices de Confusión por Experimento\n\n"

    for _, row in df.iterrows():
        md_content += f"### {row['Model']} (Semilla: {row['Seed']})\n"
        md_content += f"- **Carpeta:** `{row['Experiment']}`\n"
        md_content += f"- **Val Loss:** {row['Val_Loss']} | **Val Acc:** {row['Val_Acc']}\n"
        md_content += f"\n{row['CM']}\n\n"
        md_content += "---\n\n"

    report_path = RUNS_DIR / "global_comparison_report.md"
    with open(report_path, "w") as f:
        f.write(md_content)
        
    print(f"✅ Reporte global perfecto generado en: {report_path}")

if __name__ == "__main__":
    generate_global_report()

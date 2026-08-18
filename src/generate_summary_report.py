import json
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
RUNS_DIR = PROJECT_DIR / "runs"


def generate_global_report():
  print("=== GENERANDO REPORTE GLOBAL DEFINITIVO ===")

  records = []

  if not OUTPUT_DIR.exists():
    print(f"⚠️ No se encontró el directorio de salidas en: {OUTPUT_DIR}")
    return

  # Iterar sobre las carpetas generadas en output/
  for run_folder in sorted(OUTPUT_DIR.iterdir()):
    if run_folder.is_dir() and run_folder.name != "diagnostics":
      folder_name = run_folder.name
      config_path = run_folder / "config.json"
      summary_path = run_folder / "summary.csv"
      diag_dir = run_folder / "diagnostics"

      # 1. Inferir modelo por defecto según el nombre de la carpeta
      model_name = "Unknown"
      if (
          "PI_DEEPONET" in folder_name.upper()
          or "PIDEEPONET" in folder_name.upper()
      ):
        model_name = "PI-DeepONet"
      elif "CNN" in folder_name.upper():
        model_name = "Pure CNN"

      # 2. Inferir semilla por defecto desde el nombre de la carpeta
      seed = "N/A"
      lower_name = folder_name.lower()
      if "seed_" in lower_name:
        try:
          parts = lower_name.split("seed_")
          seed = parts[1].split("_")[0]
        except Exception:
          pass

      # 3. Refinar información leyendo config.json de forma segura (Mayúsculas/Minúsculas)
      if config_path.exists():
        try:
          with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

          # Búsqueda flexible de la clave de hiperparámetros
          hp = {}
          for key in ["Hyperparameters", "hyperparameters", "HP"]:
            if key in config:
              hp = config[key]
              break

          if "MODEL" in hp:
            model_name = hp["MODEL"]
          elif "model" in hp:
            model_name = hp["model"]

          if "SEED" in hp:
            seed = hp["SEED"]
          elif "seed" in hp:
            seed = hp["seed"]
        except Exception as e:
          print(f"⚠️ Error leyendo config.json en {folder_name}: {e}")

      # 4. Leer summary.csv para extraer métricas reales
      val_loss, val_acc = "N/A", "N/A"
      if summary_path.exists():
        try:
          df_sum = pd.read_csv(summary_path)
          if not df_sum.empty:
            last_row = df_sum.iloc[-1]
            df_sum.columns = [c.lower() for c in df_sum.columns]

            # Búsqueda de loss
            for col in ["best_val_loss", "val_loss", "v_loss", "loss"]:
              if col in df_sum.columns and pd.notna(last_row[col]):
                val_loss = f"{float(last_row[col]):.4f}"
                break

            # Búsqueda de accuracy con ajuste de escala [% o ratio [0,1]]
            for col in [
                "best_val_acc",
                "val_acc",
                "v_acc",
                "accuracy",
                "accuracy_percent",
            ]:
              if col in df_sum.columns and pd.notna(last_row[col]):
                acc_val = float(last_row[col])
                if acc_val <= 1.0:
                  val_acc = f"{acc_val * 100.0:.2f}%"
                else:
                  val_acc = f"{acc_val:.2f}%"
                break
        except Exception as e:
          print(f"⚠️ Error leyendo summary.csv en {folder_name}: {e}")

      # 5. Ubicar imagen de la matriz de confusión para el reporte Markdown
      cm_image_md = "_No disponible_"
      if diag_dir.exists():
        images = list(diag_dir.glob("confusion_matrix_*.png"))
        if images:
          img_name = images[0].name
          # Ruta relativa hacia la imagen desde output/
          cm_image_md = f"![Matriz de Confusión]({folder_name}/diagnostics/{img_name})"

      records.append({
          "Experiment": folder_name,
          "Model": model_name,
          "Seed": seed,
          "Val_Loss": val_loss,
          "Val_Acc": val_acc,
          "CM": cm_image_md,
      })

  if not records:
    print("⚠️ No se encontraron experimentos en el directorio output/.")
    return

  df = pd.DataFrame(records)

  # Construir Markdown Consolidado
  md_content = (
      "# 📊 Reporte Global de Experimentos: Comparativa y Diagnósticos\n\n"
  )
  md_content += (
      "Este reporte consolida el rendimiento de los modelos y sus respectivas"
      " matrices de confusión por corrida.\n\n"
  )
  md_content += "## 📋 Tabla Comparativa General\n\n"
  md_content += df[
      ["Experiment", "Model", "Seed", "Val_Loss", "Val_Acc"]
  ].to_markdown(index=False)
  md_content += "\n\n## 📈 Matrices de Confusión por Experimento\n\n"

  for _, row in df.iterrows():
    md_content += f"### {row['Model']} (Semilla: {row['Seed']})\n"
    md_content += f"- **Carpeta:** `{row['Experiment']}`\n"
    md_content += (
        f"- **Val Loss:** {row['Val_Loss']} | **Val Acc:** {row['Val_Acc']}\n"
    )
    md_content += f"\n{row['CM']}\n\n"
    md_content += "---\n\n"

  report_path = OUTPUT_DIR / "global_comparison_report.md"
  with open(report_path, "w", encoding="utf-8") as f:
    f.write(md_content)

  print(f"✅ Reporte global generado con éxito en: {report_path}")


if __name__ == "__main__":
  generate_global_report()

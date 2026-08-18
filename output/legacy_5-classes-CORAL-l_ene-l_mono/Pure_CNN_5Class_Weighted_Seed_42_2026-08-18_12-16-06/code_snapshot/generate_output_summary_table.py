import json
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"


def parse_diagnostics_report(report_path: Path) -> dict:
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

      if current_split in ["Val", "Test"]:
        if "macro_f1 =" in line_str:
          val = line_str.split("=")[1].strip()
          diag_metrics[f"{current_split}_Macro_F1"] = f"{float(val):.4f}"
        elif "balanced_acc =" in line_str:
          val = line_str.split("=")[1].strip()
          diag_metrics[f"{current_split}_Balanced_Acc"] = f"{float(val):.4f}"
  except Exception as e:
    print(f"⚠️ Error al parsear diagnósticos en {report_path.parent}: {e}")

  return diag_metrics


def generate_output_summary_table():
  print("=== RECOPILANDO TABLA CONSOLIDADA DE EXPERIMENTOS ===")

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

    model_name, seed = "N/A", "N/A"
    lambda_energy, lambda_mono = "N/A", "N/A"
    margin, warmup, mix_ratio = "N/A", "N/A", "N/A"

    best_val_loss, best_val_acc = "N/A", "N/A"
    test_domain, test_loss, test_acc = "N/A", "N/A", "N/A"
    val_loss_energy, val_loss_mono = "N/A", "N/A"
    duration_min, epochs = "N/A", "N/A"

    if config_path.exists():
      try:
        with open(config_path, "r", encoding="utf-8") as f:
          config_data = json.load(f)
        hp = config_data.get("Hyperparameters", {})
        model_name = hp.get("MODEL", "N/A")
        seed = hp.get("SEED", "N/A")
        lambda_energy = hp.get("LAMBDA_ENERGY", "N/A")
        lambda_mono = hp.get("LAMBDA_MONO", "N/A")
        margin = hp.get("MARGIN", "N/A")
        warmup = hp.get("WARMUP_EPOCHS", "N/A")
        mix_ratio = hp.get("MIX_RATIO", "N/A")
      except Exception as e:
        print(f"⚠️ Error leyendo config.json en {folder_name}: {e}")

    if summary_path.exists():
      try:
        df_sum = pd.read_csv(summary_path)
        if not df_sum.empty:
          row = df_sum.iloc[0]
          if "Best_Val_Loss" in df_sum.columns and pd.notna(
              row["Best_Val_Loss"]
          ):
            best_val_loss = f"{float(row['Best_Val_Loss']):.4f}"
          if "Best_Val_Acc" in df_sum.columns and pd.notna(row["Best_Val_Acc"]):
            best_val_acc = f"{float(row['Best_Val_Acc']):.2f}%"
          if "Test_Target_Domain" in df_sum.columns and pd.notna(
              row["Test_Target_Domain"]
          ):
            test_domain = str(row["Test_Target_Domain"])
          if "Test_Loss" in df_sum.columns and pd.notna(row["Test_Loss"]):
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

    if metrics_history_path.exists():
      try:
        df_mh = pd.read_csv(metrics_history_path)
        if not df_mh.empty and "val_loss" in df_mh.columns:
          best_idx = df_mh["val_loss"].idxmin()
          best_row = df_mh.loc[best_idx]
          if "val_loss_energy" in df_mh.columns and pd.notna(
              best_row["val_loss_energy"]
          ):
            val_loss_energy = f"{float(best_row['val_loss_energy']):.6f}"
          if "val_loss_mono" in df_mh.columns and pd.notna(
              best_row["val_loss_mono"]
          ):
            val_loss_mono = f"{float(best_row['val_loss_mono']):.6f}"
      except Exception as e:
        print(f"⚠️ Error leyendo metrics_history.csv en {folder_name}: {e}")

    diag_metrics = parse_diagnostics_report(diag_report_path)

    if config_path.exists() or summary_path.exists():
      records.append({
          "Experimento": folder_name,
          "Modelo": model_name,
          "L_energy": lambda_energy,
          "L_mono": lambda_mono,
          "Margin": margin,
          "Warmup": warmup,
          "MixRatio": mix_ratio,
          "Seed": seed,
          "Best_Val_Loss": best_val_loss,
          "Best_Val_Acc": best_val_acc,
          "Val_Loss_Energy": val_loss_energy,
          "Val_Loss_Mono": val_loss_mono,
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
    print("⚠️ No se encontraron carpetas válidas.")
    return

  df_out = pd.DataFrame(records)
  csv_out = OUTPUT_DIR / "tabla_resumen_experimentos.csv"
  md_out = OUTPUT_DIR / "tabla_resumen_experimentos.md"

  df_out.to_csv(csv_out, index=False)
  md_content = (
      "# 📊 Tabla Consolidada Completa de Experimentos\n\n"
      + df_out.to_markdown(index=False)
      + "\n"
  )

  with open(md_out, "w", encoding="utf-8") as f:
    f.write(md_content)

  print(f"\n✅ Proceso completado exitosamente.")


if __name__ == "__main__":
  generate_output_summary_table()

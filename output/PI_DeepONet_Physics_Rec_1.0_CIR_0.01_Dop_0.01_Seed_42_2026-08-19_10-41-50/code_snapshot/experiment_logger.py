from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
import pathlib
from pathlib import Path
import platform
import shutil
import sys
import time
import numpy as np
import pandas as pd
import torch


class ExperimentLogger:

  def __init__(self, experiment_name="PI_DeepONet_Experiment", base_dir=None):
    if base_dir is None:
      project_dir = Path(__file__).resolve().parent.parent
    else:
      project_dir = Path(base_dir).parent

    self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    self.folder_name = f"{experiment_name}_{self.timestamp}"

    # 1. Carpeta en runs/ para pesajes .pth (NO se sube a Git)
    self.run_dir = project_dir / "runs" / self.folder_name
    self.run_dir.mkdir(parents=True, exist_ok=True)

    # 2. Carpeta en output/ para métricas, gráficos y código (SÍ se sube a Git)
    self.output_dir = project_dir / "output" / self.folder_name
    self.output_dir.mkdir(parents=True, exist_ok=True)

    self.metrics_history = []
    self.start_time = None
    self.end_time = None
    self.config = {}

  def _get_sha256(self, file_path):
    file_path = Path(file_path)
    if not file_path.exists():
      return "File Not Found"
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
      for byte_block in iter(lambda: f.read(4096), b""):
        sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

  def _snapshot_code(self):
    """Copia una instantánea de TODOS los scripts (.py) en src/ incluyendo subdirectorios."""
    src_dir = Path(__file__).resolve().parent
    snapshot_dir = self.output_dir / "code_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    for py_file in src_dir.rglob("*.py"):
      relative_path = py_file.relative_to(src_dir)
      target_path = snapshot_dir / relative_path
      target_path.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(py_file, target_path)

  def get_environment_info(self):
    env_info = {
        "Fecha_Hora": self.timestamp,
        "OS": platform.system(),
        "OS_Release": platform.release(),
        "OS_Version": platform.version(),
        "Architecture": platform.machine(),
        "Python_Version": sys.version.split()[0],
        "CUDA_Available": torch.cuda.is_available(),
        "PyTorch_Version": torch.__version__,
    }

    if torch.cuda.is_available():
      env_info["GPU_Name"] = torch.cuda.get_device_name(0)
      env_info["GPU_Capability"] = (
          f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}"
      )
      env_info["GPU_VRAM_GB"] = round(
          torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
      )
      env_info["CUDA_Version"] = torch.version.cuda

    installed_packages = {}
    for dist in importlib.metadata.distributions():
      installed_packages[dist.metadata["Name"]] = dist.version

    env_info["Libraries"] = dict(sorted(installed_packages.items()))
    return env_info

  def start_experiment(self, hyperparameters, data_files):
    self.start_time = time.time()

    # Resguardo automático de los scripts de código fuente en output/
    self._snapshot_code()

    self.config = {
        "Hyperparameters": hyperparameters,
        "Data_Files": {
            str(k): {"path": str(v), "sha256": self._get_sha256(v)}
            for k, v in data_files.items()
        },
        "Environment": self.get_environment_info(),
    }

    with open(
        self.output_dir / "config.json", "w", encoding="utf-8"
    ) as f:
      json.dump(self.config, f, indent=4)

  def log_epoch(
      self,
      epoch,
      train_loss,
      train_acc,
      val_loss,
      val_acc,
      extra_metrics=None,
  ):
    entry = {
        "epoch": epoch,
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
    }
    if extra_metrics:
      entry.update(extra_metrics)

    self.metrics_history.append(entry)

  def end_experiment(self, test_metrics=None):
    self.end_time = time.time()
    total_duration_sec = (
        self.end_time - self.start_time if self.start_time else 0.0
    )

    df_metrics = pd.DataFrame(self.metrics_history)
    csv_metrics_path = self.output_dir / "metrics_history.csv"
    df_metrics.to_csv(csv_metrics_path, index=False)

    best_val_loss = (
        df_metrics["val_loss"].min()
        if not df_metrics.empty and "val_loss" in df_metrics.columns
        else None
    )
    best_val_acc = (
        df_metrics["val_acc"].max()
        if not df_metrics.empty and "val_acc" in df_metrics.columns
        else None
    )

    summary_data = {
        "Timestamp": [self.timestamp],
        "Duracion_Minutos": [round(total_duration_sec / 60.0, 2)],
        "Epocas_Completadas": [len(self.metrics_history)],
        "Best_Val_Loss": [best_val_loss],
        "Best_Val_Acc": [best_val_acc],
    }
    if test_metrics:
      for k, v in test_metrics.items():
        summary_data[f"Test_{k}"] = [v]

    df_summary = pd.DataFrame(summary_data)
    csv_summary_path = self.output_dir / "summary.csv"
    df_summary.to_csv(csv_summary_path, index=False)

    df_libs = pd.DataFrame(
        list(self.config.get("Environment", {}).get("Libraries", {}).items()),
        columns=["Library", "Version"],
    )
    csv_libs_path = self.output_dir / "environment_libraries.csv"
    df_libs.to_csv(csv_libs_path, index=False)

    self._generate_markdown_report(
        total_duration_sec, test_metrics, df_metrics
    )

    print("\n📂 Documentación y métricas livianas guardadas en:")
    print(f"   👉 {self.output_dir.resolve()}")
    print("💾 Checkpoints de modelos pesados guardados en:")
    print(f"   👉 {self.run_dir.resolve()}")

  def _generate_markdown_report(self, duration_sec, test_metrics, df_metrics):
    md_path = self.output_dir / "reporte_experimento.md"
    env = self.config.get("Environment", {})
    hp = self.config.get("Hyperparameters", {})
    files = self.config.get("Data_Files", {})

    with open(md_path, "w", encoding="utf-8") as f:
      f.write(f"# 🔬 Reporte Científico de Experimento: {self.timestamp}\n\n")

      f.write("## 1. Resumen de Ejecución\n")
      f.write(
          f"- **Duración Total:** {duration_sec / 60.0:.2f} minutos"
          f" ({duration_sec:.1f} segundos)\n"
      )
      f.write(f"- **Épocas Ejecutadas:** {len(df_metrics)}\n")
      if test_metrics:
        for k, v in test_metrics.items():
          f.write(f"- **Test {k}:** {v}\n")
      f.write("\n---\n")

      f.write("## 2. Entorno de Hardware y Software\n")
      f.write(
          f"- **Sistema Operativo:** {env.get('OS', 'N/A')}"
          f" {env.get('OS_Release', '')} ({env.get('Architecture', '')})\n"
      )
      f.write(f"- **Python Version:** {env.get('Python_Version', 'N/A')}\n")
      f.write(f"- **PyTorch Version:** {env.get('PyTorch_Version', 'N/A')}\n")
      if env.get("CUDA_Available"):
        f.write(
            f"- **GPU:** {env.get('GPU_Name')} ({env.get('GPU_VRAM_GB')} GB"
            " VRAM)\n"
        )
        f.write(f"- **CUDA Version:** {env.get('CUDA_Version')}\n")
      f.write("\n### Bibliotecas Clave Instaladas\n")
      f.write("| Biblioteca | Versión |\n| :--- | :--- |\n")
      libs = env.get("Libraries", {})
      for lib_name in [
          "torch",
          "numpy",
          "pandas",
          "scipy",
          "scikit-learn",
          "torchinfo",
          "torchviz",
          "onnx",
      ]:
        if lib_name in libs:
          f.write(f"| `{lib_name}` | `{libs[lib_name]}` |\n")
      f.write(
          f"\n*(Ver lista completa de {len(libs)} bibliotecas en"
          " `environment_libraries.csv`)*\n\n---\n"
      )

      f.write("## 3. Configuración e Hiperparámetros\n")
      f.write("| Parámetro | Valor |\n| :--- | :--- |\n")
      for k, v in hp.items():
        f.write(f"| `{k}` | `{v}` |\n")
      f.write("\n---\n")

      f.write("## 4. Trazabilidad de Archivos de Datos (Hashes SHA-256)\n")
      f.write(
          "| Identificador | Ruta | SHA-256 Hash |\n| :--- | :--- | :--- |\n"
      )
      for k, v in files.items():
        f.write(
            f"| `{k}` | `{v.get('path')}` | `{v.get('sha256', '')[:16]}...` |\n"
        )
      f.write("\n---\n")

      f.write("## 5. Historial de Entrenamiento por Época\n")
      if not df_metrics.empty:
        f.write(df_metrics.to_markdown(index=False))
      else:
        f.write("_No hay métricas registradas._")
      f.write("\n")

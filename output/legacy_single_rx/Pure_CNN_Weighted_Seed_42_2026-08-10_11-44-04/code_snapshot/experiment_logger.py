import os
import sys
import platform
import json
import hashlib
import time
import shutil
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import importlib.metadata

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

        # 2. Carpeta en outputs/ para métricas, gráficos y código (SÍ se sube a Git)
        self.output_dir = project_dir / "outputs" / self.folder_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history = []
        self.start_time = None
        self.end_time = None

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
        """Copia una instantánea de todos los programas de Python en src/ hacia outputs/"""
        src_dir = Path(__file__).resolve().parent
        snapshot_dir = self.output_dir / "code_snapshot"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        for py_file in src_dir.glob("*.py"):
            shutil.copy2(py_file, snapshot_dir / py_file.name)

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
            env_info["GPU_Capability"] = f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}"
            env_info["GPU_VRAM_GB"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            env_info["CUDA_Version"] = torch.version.cuda

        installed_packages = {}
        for dist in importlib.metadata.distributions():
            installed_packages[dist.metadata["Name"]] = dist.version

        env_info["Libraries"] = dict(sorted(installed_packages.items()))
        return env_info

    def start_experiment(self, hyperparameters, data_files):
        self.start_time = time.time()

        # Resguardo automático de los scripts de código fuente en outputs/
        self._snapshot_code()

        self.config = {
            "Hyperparameters": hyperparameters,
            "Data_Files": {str(k): {"path": str(v), "sha256": self._get_sha256(v)} for k, v in data_files.items()},
            "Environment": self.get_environment_info()
        }

        with open(self.output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4)

    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, extra_metrics=None):
        entry = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc)
        }
        if extra_metrics:
            entry.update(extra_metrics)

        self.metrics_history.append(entry)

    def end_experiment(self, test_metrics=None):
        self.end_time = time.time()
        total_duration_sec = self.end_time - self.start_time

        df_metrics = pd.DataFrame(self.metrics_history)
        csv_metrics_path = self.output_dir / "metrics_history.csv"
        df_metrics.to_csv(csv_metrics_path, index=False)

        summary_data = {
            "Timestamp": [self.timestamp],
            "Duracion_Minutos": [round(total_duration_sec / 60.0, 2)],
            "Epocas_Completadas": [len(self.metrics_history)],
            "Best_Val_Loss": [df_metrics["val_loss"].min() if not df_metrics.empty else None],
            "Best_Val_Acc": [df_metrics["val_acc"].max() if not df_metrics.empty else None],
        }
        if test_metrics:
            for k, v in test_metrics.items():
                summary_data[f"Test_{k}"] = [v]

        df_summary = pd.DataFrame(summary_data)
        csv_summary_path = self.output_dir / "summary.csv"
        df_summary.to_csv(csv_summary_path, index=False)

        df_libs = pd.DataFrame(
            list(self.config["Environment"]["Libraries"].items()),
            columns=["Library", "Version"]
        )
        csv_libs_path = self.output_dir / "environment_libraries.csv"
        df_libs.to_csv(csv_libs_path, index=False)

        self._generate_markdown_report(total_duration_sec, test_metrics, df_metrics)

        print(f"\n📂 Documentación y métricas livianas guardadas en:")
        print(f"  👉 {self.output_dir.resolve()}")
        print(f"💾 Checkpoints de modelos pesados guardados en:")
        print(f"  👉 {self.run_dir.resolve()}")

    def _generate_markdown_report(self, duration_sec, test_metrics, df_metrics):
        md_path = self.output_dir / "reporte_experimento.md"
        env = self.config["Environment"]
        hp = self.config["Hyperparameters"]
        files = self.config["Data_Files"]

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# 🔬 Reporte Científico de Experimento: {self.timestamp}\n\n")

            f.write("## 1. Resumen de Ejecución\n")
            f.write(f"- **Duración Total:** {duration_sec / 60.0:.2f} minutos ({duration_sec:.1f} segundos)\n")
            f.write(f"- **Épocas Ejecutadas:** {len(df_metrics)}\n")
            if test_metrics:
                for k, v in test_metrics.items():
                    f.write(f"- **Test {k}:** {v}\n")
            f.write("\n---\n")

            f.write("## 2. Entorno de Hardware y Software\n")
            f.write(f"- **Sistema Operativo:** {env['OS']} {env['OS_Release']} ({env['Architecture']})\n")
            f.write(f"- **Python Version:** {env['Python_Version']}\n")
            f.write(f"- **PyTorch Version:** {env['PyTorch_Version']}\n")
            if env['CUDA_Available']:
                f.write(f"- **GPU:** {env['GPU_Name']} ({env['GPU_VRAM_GB']} GB VRAM)\n")
                f.write(f"- **CUDA Version:** {env['CUDA_Version']}\n")
            f.write("\n### Bibliotecas Clave Instaladas\n")
            f.write("| Biblioteca | Versión |\n| :--- | :--- |\n")
            for lib_name in ["torch", "numpy", "pandas", "scipy", "scikit-learn", "torchinfo", "torchviz", "onnx"]:
                if lib_name in env["Libraries"]:
                    f.write(f"| `{lib_name}` | `{env['Libraries'][lib_name]}` |\n")
            f.write(f"\n*(Ver lista completa de {len(env['Libraries'])} bibliotecas en `environment_libraries.csv`)*\n\n---\n")

            f.write("## 3. Configuración e Hiperparámetros\n")
            f.write("| Parámetro | Valor |\n| :--- | :--- |\n")
            for k, v in hp.items():
                f.write(f"| `{k}` | `{v}` |\n")
            f.write("\n---\n")

            f.write("## 4. Trazabilidad de Archivos de Datos (Hashes SHA-256)\n")
            f.write("| Identificador | Ruta | SHA-256 Hash |\n| :--- | :--- | :--- |\n")
            for k, v in files.items():
                f.write(f"| `{k}` | `{v['path']}` | `{v['sha256'][:16]}...` |\n")
            f.write("\n---\n")

            f.write("## 5. Historial de Entrenamiento por Época\n")
            f.write(df_metrics.to_markdown(index=False))
            f.write("\n")

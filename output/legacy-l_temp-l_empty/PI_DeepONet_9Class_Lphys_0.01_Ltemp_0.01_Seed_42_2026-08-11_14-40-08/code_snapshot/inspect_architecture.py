import os
from pathlib import Path
import torch
import numpy as np

# Importar el modelo desde el módulo 5b
from m5b_pideeponet_model import PIDeepONet

# Bibliotecas de visualización
from torchinfo import summary
import torchviz
import onnx

# ==============================================================================
# CONFIGURACIÓN Y RUTAS DE SALIDA
# ==============================================================================
OUTPUT_DIR = Path(r"C:\Users\fdaniellc\Desktop\PI-DeepONet\project\architecture_inspection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inspect_network():
    print("=== INSPECCIÓN COMPLETA DE ARQUITECTURA PI-DEEPONET ===")
    print(f"Dispositivo: {DEVICE}\n")

    # 1. Instanciar Modelo con 9 clases de conteo
    model = PIDeepONet(num_classes=9, latent_dim=128).to(DEVICE)
    model.eval()

    # Tensor de prueba: (Batch_Size=1, Channels=2, Time=25, Subcarriers=241)
    dummy_input = torch.randn(1, 2, 25, 241).to(DEVICE)

    # --------------------------------------------------------------------------
    # OPCIÓN 1: TABLA DETALLADA CON `torchinfo`
    # --------------------------------------------------------------------------
    print("1. Generando resumen tabular con 'torchinfo'...")
    info_str = str(summary(
        model, 
        input_size=(1, 2, 25, 241),
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        depth=4,
        verbose=0
    ))
    
    print(info_str)
    
    # Guardar resumen en archivo de texto
    with open(OUTPUT_DIR / "pideeponet_torchinfo_summary.txt", "w", encoding="utf-8") as f:
        f.write(info_str)
    print(f"  [OK] Resumen guardado en: {OUTPUT_DIR / 'pideeponet_torchinfo_summary.txt'}\n")

# --------------------------------------------------------------------------
    # OPCIÓN 2: GRAFO COMPUTACIONAL CON `torchviz`
    # --------------------------------------------------------------------------
    print("2. Generando grafo de autograd con 'torchviz'...")
    try:
        logits, latent_field = model(dummy_input)
        
        # Grafo del flujo de operaciones
        graph = torchviz.make_dot(
            (logits, latent_field), 
            params=dict(model.named_parameters()),
            show_attrs=True,
            show_saved=True
        )
        
        # Cambiamos formato a 'png' (o 'svg') para evitar el fallo de pango.dll en Windows
        graph_path = OUTPUT_DIR / "pideeponet_computational_graph"
        graph.render(filename=str(graph_path), format="png", cleanup=True)
        print(f"  [OK] Grafo computacional exportado a imagen: {graph_path}.png\n")
    except Exception as e:
        print(f"  [ADVERTENCIA] No se pudo generar el grafo de torchviz: {e}\n")

    # --------------------------------------------------------------------------
    # OPCIÓN 3: EXPORTACIÓN ONNX PARA VISUALIZACIÓN INTERACTIVA CON NETRON
    # --------------------------------------------------------------------------
    print("3. Exportando modelo a formato ONNX (para Netron)...")
    onnx_path = OUTPUT_DIR / "pideeponet.onnx"
    
    # Exportar grafo de PyTorch a ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['CSI_Input_Branch'],
        output_names=['Logits_Counting', 'Latent_Field'],
        dynamic_axes={
            'CSI_Input_Branch': {0: 'batch_size'},
            'Logits_Counting': {0: 'batch_size'},
            'Latent_Field': {0: 'batch_size'}
        }
    )
    
    # Validar el archivo ONNX generado
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"  [OK] Modelo ONNX verificado y guardado en: {onnx_path}")
    print("\n" + "="*70)
    print("💡 INSTRUCCIÓN PARA NETRON:")
    print("1. Abre tu navegador y entra a: https://netron.app")
    print(f"2. Arrastra y suelta el archivo '{onnx_path}'")
    print("3. Podrás hacer clic en cada nodo para inspeccionar matrices, capas y dimensiones.")
    print("="*70)

if __name__ == "__main__":
    inspect_network()
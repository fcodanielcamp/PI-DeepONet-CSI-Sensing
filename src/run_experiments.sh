#!/bin/bash

# Semilla única fijada para comparativa justa
SEED=42

echo "================================================================="
echo " 🚀 INICIANDO BATERÍA DE EXPERIMENTOS (5 CLASES): COMPOSICIÓN Y COMPARATIVA"
echo "================================================================="

# 1/3: PI-DeepONet con Regularización Física Integrada
echo ""
echo "=== [1/3] Ejecutando PI-DeepONet con Física (L_energy=0.01, L_mono=0.001) ==="
python3 6_train_pideeponet.py --seed "$SEED" --lambda_energy 0.01 --lambda_mono 0.001 --margin 0.0 --warmup_epochs 5 --mix_ratio 0.0

# 2/3: DeepONet Data-Driven Puro (Pérdidas físicas desactivadas)
echo ""
echo "=== [2/3] Ejecutando DeepONet Pure Data-Driven (L_energy=0.0, L_mono=0.0) ==="
python3 6_train_pideeponet.py --seed "$SEED" --lambda_energy 0.0 --lambda_mono 0.0 --margin 0.0 --warmup_epochs 0 --mix_ratio 0.0

# 3/3: Pure CNN Baseline
echo ""
echo "=== [3/3] Ejecutando Pure CNN2D Baseline ==="
python3 6c_train_pure_cnn.py --seed "$SEED"

# Reporte Global Consolidado
echo ""
echo "=== Generando Reporte Comparativo Global Markdown ==="
python3 generate_summary_report.py

echo ""
echo "✅ ¡Experimentos completados y reporte consolidado generado con éxito!"

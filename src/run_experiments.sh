#!/bin/bash

# Terminar si ocurre un error
set -e

SEED=42

echo "================================================================="
echo " 🚀 BATERÍA DE EXPERIMENTOS CONSENSUS: ABLACIÓN FÍSICA (5 CLASES)"
echo "================================================================="

# Experimento 1 (Control): Baseline Data-Driven Puro
# Solo loss supervisada (CORAL), pérdidas físicas en 0.0
echo ""
echo "=== [1/3] Exp 1: Baseline Data-Driven (Solo L_CORAL) ==="
python3 6_train_pideeponet.py \
    --seed "$SEED" \
    --lambda_energy 0.0 \
    --lambda_mono 0.0 \
    --lambda_rec 0.0 \
    --lambda_cir 0.0 \
    --lambda_dop 0.0 \
    --margin 0.0 \
    --warmup_epochs 0 \
    --mix_ratio 0.0

# Experimento 2 (Ablación de Representación): Latente Físico sin Restricciones
# Solo reconstrucción activa (L_rec = 1.0), sin regularizaciones físicas
echo ""
echo "=== [2/3] Exp 2: Latente Físico sin Restricciones (L_CORAL + L_rec) ==="
python3 6_train_pideeponet.py \
    --seed "$SEED" \
    --lambda_energy 0.0 \
    --lambda_mono 0.0 \
    --lambda_rec 1.0 \
    --lambda_cir 0.0 \
    --lambda_dop 0.0 \
    --margin 0.0 \
    --warmup_epochs 0 \
    --mix_ratio 0.0

# Experimento 3 (Intervención Física): Physics-Informed Completo
# Todas las pérdidas físicas y de reconstrucción activas con warmup
echo ""
echo "=== [3/3] Exp 3: Physics-Informed Completo (Física e Interferometría) ==="
python3 6_train_pideeponet.py \
    --seed "$SEED" \
    --lambda_energy 0.01 \
    --lambda_mono 0.001 \
    --lambda_rec 1.0 \
    --lambda_cir 0.01 \
    --lambda_dop 0.01 \
    --margin 0.0 \
    --warmup_epochs 5 \
    --mix_ratio 0.0

# Reporte Global Consolidado y Diagnósticos
echo ""
echo "=== Generando Reporte Comparativo Global y Diagnósticos Markdown ==="
python3 generate_summary_report.py

echo ""
echo "✅ ¡Secuencia de 3 experimentos completada con éxito!"

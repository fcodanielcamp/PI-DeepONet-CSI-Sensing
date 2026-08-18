#!/bin/bash

# Terminar la ejecución si ocurre algún error en la cadena
set -e

SEED=42

echo "================================================================="
echo " 🚀 BATERÍA DE EXPERIMENTOS CONSENSUS: ABLACIÓN FÍSICA (5 CLASES)"
echo "================================================================="

# Experimento 1 (Control): Baseline Data-Driven Puro
# Latente abstracto (128), solo loss CORAL activa
echo ""
echo "=== [1/3] Exp 1: Baseline Data-Driven (Latente abstracto 128, solo L_CORAL) ==="
python3 6_train_pideeponet.py \
    --seed "$SEED" \
    --use_physics_latent 0 \
    --latent_dim 128 \
    --lambda_rec 0.0 \
    --lambda_cir 0.0 \
    --lambda_dd 0.0 \
    --lambda_smooth 0.0 \
    --lambda_sparse 0.0 \
    --warmup_epochs 0

# Experimento 2 (Ablación de Representación): Latente Físico sin Restricciones
# Ramas CIR (64) + Doppler (64), activa loss de reconstrucción (L_rec = 1.0)
echo ""
echo "=== [2/3] Exp 2: Latente Físico sin Restricciones (CIR+Doppler, L_CORAL + L_rec) ==="
python3 6_train_pideeponet.py \
    --seed "$SEED" \
    --use_physics_latent 1 \
    --latent_dim 128 \
    --lambda_rec 1.0 \
    --lambda_cir 0.0 \
    --lambda_dd 0.0 \
    --lambda_smooth 0.0 \
    --lambda_sparse 0.0 \
    --warmup_epochs 0

# Experimento 3 (Intervención Física): Physics-Informed Completo
# Ramas CIR (64) + Doppler (64), activa todas las losses físicas + Warmup por etapas
echo ""
echo "=== [3/3] Exp 3: Physics-Informed Completo (CIR+Doppler, Todas las Pérdidas Físicas) ==="
python3 6_train_pideeponet.py \
    --seed "$SEED" \
    --use_physics_latent 1 \
    --latent_dim 128 \
    --lambda_rec 1.0 \
    --lambda_cir 0.01 \
    --lambda_dd 0.01 \
    --lambda_smooth 0.001 \
    --lambda_sparse 0.001 \
    --warmup_epochs 5

# Reporte Global Consolidado y Diagnósticos
echo ""
echo "=== Generando Reporte Comparativo Global y Diagnósticos Markdown ==="
python3 generate_summary_report.py

echo ""
echo "✅ ¡Secuencia de 3 experimentos completada con éxito!"

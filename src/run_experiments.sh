#!/bin/bash

# Semilla única fijada para el experimento diagnóstico
SEED=42

# 1. Entrenar PI-DeepONet (CORAL + Física Dirichlet + Temporal + Energía RCS)
echo "=== Ejecutando PI-DeepONet (CORAL + Multi-Física) con semilla $SEED ==="
python3 6_train_pideeponet.py --seed "$SEED" --lambda_phys 0.01 --lambda_temp 0.001 --lambda_energy 0.001 --gamma 1.0

# 2. Entrenar Pure CNN Baseline (con el conjunto de tensores actual)
echo "=== Ejecutando Pure CNN Baseline con semilla $SEED ==="
python3 6c_train_pure_cnn.py --seed "$SEED"

echo "✅ ¡Pipeline completo de experimentos ejecutado con éxito!"

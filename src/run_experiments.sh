#!/bin/bash

# Semilla única fijada para el experimento diagnóstico
SEED=42

# 1. Entrenar PI-DeepONet (CORAL + Física Dirichlet + Temporal + Energía RCS)
echo "=== Ejecutando PI-DeepONet (CORAL + Multi-Física) con semilla $SEED ==="
python3 6_train_pideeponet.py --seed "$SEED" --lambda_phys 0.1 --lambda_temp 0.0 --lambda_energy 0.0 --gamma 0.0

echo "=== Ejecutando 2/5 ==="
python3 6_train_pideeponet.py --seed "$SEED" --lambda_phys 0.0 --lambda_temp 0.1 --lambda_energy 0.0 --gamma 0.0

echo "=== Ejecutando 3/5 ==="
python3 6_train_pideeponet.py --seed "$SEED" --lambda_phys 0.0 --lambda_temp 0.0 --lambda_energy 0.1 --gamma 1.0

echo "=== Ejecutando 4/5 ==="
python3 6_train_pideeponet.py --seed "$SEED" --lambda_phys 0.0 --lambda_temp 0.0 --lambda_energy 0.01 --gamma 2.0

echo "=== Ejecutando 5/5 ==="
python3 6_train_pideeponet.py --seed "$SEED" --lambda_phys 0.1 --lambda_temp 0.1 --lambda_energy 0.1 --gamma 1.0

echo "✅ ¡Pipeline completo de experimentos ejecutado con éxito!"

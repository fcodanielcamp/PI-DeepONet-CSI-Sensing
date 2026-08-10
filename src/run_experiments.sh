#!/bin/bash

# Semilla única fijada para el experimento diagnóstico
SEED=42

# 1. Entrenar PI-DeepONet (3 clases)
echo "=== Ejecutando PI-DeepONet (3 clases) con semilla $SEED ==="
python3 6_train_pideeponet.py --seed "$SEED"

# 2. Entrenar Pure CNN Baseline (3 clases)
echo "=== Ejecutando Pure CNN (3 clases) con semilla $SEED ==="
python3 6c_train_pure_cnn.py --seed "$SEED"

echo "✅ ¡Experimento diagnóstico puente (3 clases) completado con éxito!"

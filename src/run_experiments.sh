#!/bin/bash

# Semilla única fijada para el benchmark de monotonicidad
SEED=42

echo "================================================================="
echo " 🚀 INICIANDO BATERÍA DE EXPERIMENTOS: MONOTONICIDAD SUAVE"
echo "================================================================="

# 1/5: Control Monótono (Referencia limpia)
echo ""
echo "=== [1/5] Ejecutando Control Monótono (L_mono=0.001, Mix=0.0) ==="
python3 6_train_pideeponet.py --seed "$SEED" --lambda_energy 0.01 --lambda_mono 0.001 --margin 0.0 --warmup_epochs 5 --mix_ratio 0.0

# 2/5: Puntos Mejor Elegidos (70% Vecinos + 30% Mezclas)
echo ""
echo "=== [2/5] Ejecutando Puntos Mejor Elegidos (Mix 70/30) ==="
python3 6_train_pideeponet.py --seed "$SEED" --lambda_energy 0.01 --lambda_mono 0.001 --margin 0.0 --warmup_epochs 5 --mix_ratio 0.3

# 3/5: Monotonicidad Más Fuerte
echo ""
echo "=== [3/5] Ejecutando Monotonicidad Más Fuerte (L_mono=0.003) ==="
python3 6_train_pideeponet.py --seed "$SEED" --lambda_energy 0.01 --lambda_mono 0.003 --margin 0.0 --warmup_epochs 5 --mix_ratio 0.0

# 4/5: Energía Alta
echo ""
echo "=== [4/5] Ejecutando Energía Alta (L_energy=0.1) ==="
python3 6_train_pideeponet.py --seed "$SEED" --lambda_energy 0.1 --lambda_mono 0.001 --margin 0.0 --warmup_epochs 5 --mix_ratio 0.0

# 5/5: Versión Conservadora
echo ""
echo "=== [5/5] Ejecutando Versión Conservadora (Margin=0.01, Warmup=8, Mix=0.3) ==="
python3 6_train_pideeponet.py --seed "$SEED" --lambda_energy 0.01 --lambda_mono 0.001 --margin 0.01 --warmup_epochs 8 --mix_ratio 0.3

# Consolidador final de la tabla
echo ""
echo "=== Consolidador: Generando Tabla Resumen Final ==="
python3 generate_output_summary_table.py

echo ""
echo "✅ ¡Pipeline de 5 experimentos completado y registrado con éxito!"

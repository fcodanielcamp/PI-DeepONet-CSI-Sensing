#!/bin/bash
# ==============================================================================
# BATERÍA DE EXPERIMENTOS ENRIQUECIDA -- v2
#
# Diferencias frente a run_experiments.sh original:
#   1. Repite la ablación (Exp1/2/3) con la corrección LayerNorm(128) recién
#      aplicada en BranchNet (m5b_pideeponet_model.py).
#   2. Corre cada experimento con 3 semillas distintas (42, 123, 777) para
#      robustez estadística -- una sola semilla no es defendible para el paper.
#   3. Corrige el bug de checkpoints sobrescritos: best_model_path en
#      6_train_pideeponet.py depende SOLO de la semilla, no de los lambdas,
#      así que cada corrida pisaba el checkpoint de la anterior. Aquí se
#      renombra inmediatamente después de cada entrenamiento.
#   4. Ejecuta automáticamente diagnose_classifier_head_shift.py al final de
#      cada corrida, guardando el log -- así verificamos si LayerNorm recuperó
#      la correlación clase-real/escalar-proyectado en test (estaba en 0.080
#      antes del fix) sin tener que correrlo a mano por cada checkpoint.
#   5. NO usa "set -e" a nivel global: si una corrida individual falla, se
#      registra el error y la batería continúa con la siguiente, en vez de
#      abortar todo el proceso (relevante para dejarlo corriendo desatendido
#      varias horas).
#
# Duración estimada: 9 entrenamientos completos (~85 min c/u según corridas
# previas) + diagnósticos ligeros -> aprox. 13 horas.
# ==============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="$PROJECT_DIR/../models"
DIAG_LOG_DIR="$PROJECT_DIR/../diagnostic_logs"
mkdir -p "$DIAG_LOG_DIR"

SEEDS=(42 123 777)

# Definición de los 3 experimentos de la ablación física (idéntica a la original)
declare -a EXP_NAMES=(
  "Exp1_baseline_data_driven"
  "Exp2_reconstruccion_sin_restricciones"
  "Exp3_physics_informed_completo"
)
declare -a EXP_ARGS=(
  "--lambda_energy 0.0  --lambda_mono 0.0   --lambda_rec 0.0 --lambda_cir 0.0  --lambda_dop 0.0  --margin 0.0 --warmup_epochs 0 --mix_ratio 0.0"
  "--lambda_energy 0.0  --lambda_mono 0.0   --lambda_rec 1.0 --lambda_cir 0.0  --lambda_dop 0.0  --margin 0.0 --warmup_epochs 0 --mix_ratio 0.0"
  "--lambda_energy 0.01 --lambda_mono 0.001 --lambda_rec 1.0 --lambda_cir 0.01 --lambda_dop 0.01 --margin 0.0 --warmup_epochs 5 --mix_ratio 0.0"
)

FAILED_RUNS=()
TOTAL_RUNS=$((${#SEEDS[@]} * ${#EXP_NAMES[@]}))
RUN_COUNT=0

echo "================================================================="
echo " 🚀 BATERÍA ENRIQUECIDA: ABLACIÓN FÍSICA x MULTI-SEED (post-LayerNorm)"
echo " Total de corridas programadas: $TOTAL_RUNS"
echo "================================================================="

for SEED in "${SEEDS[@]}"; do
  for i in "${!EXP_NAMES[@]}"; do
    RUN_COUNT=$((RUN_COUNT + 1))
    EXP_NAME="${EXP_NAMES[$i]}"
    ARGS="${EXP_ARGS[$i]}"
    RUN_TAG="${EXP_NAME}_seed${SEED}"

    echo ""
    echo "-----------------------------------------------------------------"
    echo "[$RUN_COUNT/$TOTAL_RUNS] $RUN_TAG  |  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "-----------------------------------------------------------------"

    # shellcheck disable=SC2086
    if python3 6_train_pideeponet.py --seed "$SEED" $ARGS; then

      # El checkpoint que acaba de generarse SIEMPRE se llama igual
      # (depende solo de la semilla) -- lo renombramos ya mismo para que la
      # siguiente corrida con la misma semilla no lo sobrescriba.
      SRC_CKPT="$CHECKPOINT_DIR/pideeponet_physics_seed_${SEED}_best.pth"
      DST_CKPT="$CHECKPOINT_DIR/pideeponet_${RUN_TAG}_best.pth"

      if [ -f "$SRC_CKPT" ]; then
        cp "$SRC_CKPT" "$DST_CKPT"
        echo "  ✅ Checkpoint preservado en: $DST_CKPT"

        # Diagnóstico automático de correlación clase-real/escalar-proyectado.
        # Verifica si LayerNorm recuperó la señal perdida bajo domain-shift.
        DIAG_LOG="$DIAG_LOG_DIR/diag_${RUN_TAG}.log"
        echo "  🔎 Corriendo diagnóstico de cabezal CORAL -> $DIAG_LOG"
        if python3 diagnose_classifier_head_shift.py --model_path "$DST_CKPT" > "$DIAG_LOG" 2>&1; then
          CORR_TEST=$(grep "Correlación (clase real, projected) en TEST" "$DIAG_LOG" | tail -1)
          echo "  📊 $CORR_TEST"
        else
          echo "  ⚠️ El diagnóstico falló para $RUN_TAG -- revisar $DIAG_LOG"
        fi
      else
        echo "  ⚠️ No se encontró el checkpoint esperado en $SRC_CKPT -- no se pudo diagnosticar."
        FAILED_RUNS+=("$RUN_TAG (checkpoint no encontrado)")
      fi

    else
      echo "  ❌ FALLÓ el entrenamiento de $RUN_TAG -- continuando con la siguiente corrida."
      FAILED_RUNS+=("$RUN_TAG (entrenamiento falló)")
    fi
  done
done

echo ""
echo "================================================================="
echo " 📋 RESUMEN DE LA BATERÍA"
echo "================================================================="
echo " Corridas completadas: $((TOTAL_RUNS - ${#FAILED_RUNS[@]})) / $TOTAL_RUNS"

if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
  echo " Corridas con problemas:"
  for f in "${FAILED_RUNS[@]}"; do
    echo "   - $f"
  done
else
  echo " ✅ Todas las corridas se completaron sin errores."
fi

echo ""
echo "=== Generando reportes consolidados ==="
python3 generate_summary_report.py || echo "⚠️ generate_summary_report.py falló -- revisar manualmente."
python3 generate_output_summary_table.py || echo "⚠️ generate_output_summary_table.py falló -- revisar manualmente."

echo ""
echo "✅ Batería enriquecida finalizada: $(date '+%Y-%m-%d %H:%M:%S')"
echo "   Diagnósticos de cabezal por corrida en: $DIAG_LOG_DIR/"
echo "   Checkpoints preservados en:             $CHECKPOINT_DIR/"

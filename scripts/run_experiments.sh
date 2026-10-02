#!/usr/bin/env bash
# Experimentos de la Pregunta 1 (E2 modelos, E4 ablaciones, E6 desbalance). Requiere GPU.
# Cada experimento sigue aunque otro falle; al final se resume todo.
cd "$(dirname "$0")/.."
EXTRA="$*"   # p. ej. --epochs 1 --limit 20 para una prueba rápida

python scripts/07_train.py --model scratch                --run_name A_scratch_bce_soft     $EXTRA
python scripts/07_train.py --model resnet50               --run_name B_resnet50_bce_soft    $EXTRA
python scripts/07_train.py --model resnet50 --loss focal  --run_name C_resnet50_focal_soft  $EXTRA
python scripts/07_train.py --model resnet50 --labels hard --run_name D_resnet50_bce_hard    $EXTRA
python scripts/07_train.py --model resnet50 --pos_weight none --run_name E_resnet50_bce_soft_nopw $EXTRA
python scripts/07_train.py --model resnet50 --loss focal --plane Coronal --run_name F_resnet50_focal_soft_cor $EXTRA
python scripts/08_summarize.py

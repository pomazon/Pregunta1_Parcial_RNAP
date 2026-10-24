#!/usr/bin/env bash
# Comando único del proyecto (Pregunta 1).
#
#   bash run_all.sh [--quick] [DATA_DIR]
#
#   (sin opciones)  corrida completa: reproduce los resultados del informe.
#                   Kaggle, GPU T4, ~varias horas (traducción + 6 entrenamientos).
#   --quick         prueba de humo de extremo a extremo (~15-25 min): todos los pasos con
#                   pocos estudios y 1 epoch. Comprueba que el pipeline corre; sus métricas
#                   NO son resultados. Usar en un notebook limpio (escribe en outputs/).
#
# DATA_DIR: carpeta de la competencia (por defecto, la ruta estándar de Kaggle).
set -euo pipefail

QUICK=0
DATA_DIR="/kaggle/input/competitions/rsna-knee-abnormality-detection"
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    *) DATA_DIR="$arg" ;;
  esac
done
cd "$(dirname "$0")"

if [ "$QUICK" -eq 1 ]; then
  echo "### MODO RÁPIDO: prueba de humo, las métricas no son resultados ###"
  TRANSLATE_ARGS="--limit 200"                 # los 58 gold + 142 informes más
  SUBSET_ARGS="--n_studies 80 --min_pos 3"
  PREPROCESS_ARGS="--limit 60"                 # 20 estudios por partición
  TRAIN_ARGS="--epochs 1 --limit 20 --workers 2"
else
  TRANSLATE_ARGS="" SUBSET_ARGS="" PREPROCESS_ARGS="" TRAIN_ARGS=""
fi
START=$(date +%s)

pip install -q -r requirements.txt

echo "== Paso 1: EDA =="
python scripts/01_eda.py --data_dir "$DATA_DIR" --out_dir outputs/eda

echo "== Paso 2: traducción de reportes al inglés (GPU; se omite si ya existe) =="
python scripts/02_translate_reports.py --data_dir "$DATA_DIR" $TRANSLATE_ARGS

echo "== Paso 3: extracción de etiquetas débiles =="
python scripts/03_extract_labels.py

echo "== Paso 4: validación del extractor contra gold =="
python scripts/04_validate_labels.py --data_dir "$DATA_DIR"

echo "== Paso 5: subconjunto y particiones =="
python scripts/05_build_subset.py --data_dir "$DATA_DIR" $SUBSET_ARGS

echo "== Paso 6: preprocesamiento DICOM -> caché =="
python scripts/06_preprocess.py --data_dir "$DATA_DIR" $PREPROCESS_ARGS

echo "== Pasos 7-8: entrenamiento, ablaciones y resumen (GPU) =="
bash scripts/run_experiments.sh $TRAIN_ARGS

echo "== Paso 9: Grad-CAM y atención (mejor modelo) =="
python scripts/09_gradcam.py --run C_resnet50_focal_soft

echo "== Paso 10: figuras y tablas del informe =="
python scripts/10_figures.py

echo "== Listo en $(( ($(date +%s) - START) / 60 )) min. Resultados en outputs/ =="

#!/usr/bin/env bash
# Comando único del proyecto: bash run_all.sh [DATA_DIR]
# Cada paso se añade aquí a medida que se implementa.
set -euo pipefail
DATA_DIR="${1:-/kaggle/input/competitions/rsna-knee-abnormality-detection}"
cd "$(dirname "$0")"

pip install -q -r requirements.txt

echo "== Paso 1: EDA =="
python scripts/01_eda.py --data_dir "$DATA_DIR" --out_dir outputs/eda

echo "== Paso 2: traducción de reportes al inglés (GPU; se omite si ya existe) =="
python scripts/02_translate_reports.py --data_dir "$DATA_DIR"

echo "== Paso 3: extracción de etiquetas débiles =="
python scripts/03_extract_labels.py

echo "== Paso 4: validación del extractor contra gold =="
python scripts/04_validate_labels.py --data_dir "$DATA_DIR"

echo "== Paso 5: subconjunto y particiones =="
python scripts/05_build_subset.py --data_dir "$DATA_DIR"

echo "== Paso 6: preprocesamiento DICOM -> caché =="
python scripts/06_preprocess.py --data_dir "$DATA_DIR"

echo "== Pasos 7-8: entrenamiento, ablaciones y resumen (GPU) =="
bash scripts/run_experiments.sh

# Próximo paso:
# python scripts/09_gradcam.py

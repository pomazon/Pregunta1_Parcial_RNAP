#!/usr/bin/env bash
# Comando único del proyecto: bash run_all.sh [DATA_DIR]
# Cada paso se añade aquí a medida que se implementa.
set -euo pipefail
DATA_DIR="${1:-/kaggle/input/rsna-knee-abnormality-detection}"
cd "$(dirname "$0")"

pip install -q -r requirements.txt

echo "== Paso 1: EDA =="
python scripts/01_eda.py --data_dir "$DATA_DIR" --out_dir outputs/eda

# Próximos pasos (se irán descomentando):
# python scripts/02_extract_labels.py   --data_dir "$DATA_DIR"
# python scripts/03_validate_labels.py  --data_dir "$DATA_DIR"
# python scripts/04_build_subset.py     --data_dir "$DATA_DIR"
# python scripts/05_preprocess.py       --data_dir "$DATA_DIR"
# python scripts/06_train.py            --model scratch
# python scripts/06_train.py            --model resnet50
# python scripts/07_evaluate.py
# python scripts/08_gradcam.py

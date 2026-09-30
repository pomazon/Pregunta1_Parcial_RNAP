"""Utilidades compartidas por los scripts del pipeline."""
import os

DEFAULT_DATA_DIR = "/kaggle/input/competitions/rsna-knee-abnormality-detection"


def resolve_data_dir(path):
    """Devuelve la carpeta que contiene train.csv y train_series.csv.

    Si `path` no los contiene, los busca bajo /kaggle/input
    (Kaggle puede montar la competencia en rutas distintas).
    """
    if os.path.isfile(os.path.join(path, "train_series.csv")):
        return path
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train_series.csv" in files and "train.csv" in files:
            print(f"AVISO: datos no encontrados en {path}; se usan los de {root}")
            return root
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
    raise FileNotFoundError(
        f"No se encontró train_series.csv en {path} ni bajo /kaggle/input. "
        "¿Está añadida la competencia como Input?")


def detect_language(text):
    """Código ISO 639-1 del idioma del reporte ('na' si es muy corto)."""
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0
    try:
        return detect(text) if isinstance(text, str) and len(text.strip()) > 20 else "na"
    except Exception:
        return "err"

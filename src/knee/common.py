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


def find_file(name, local_path, search_root="/kaggle/input"):
    """Ruta de un artefacto del pipeline.

    Usa `local_path` si existe (ejecución completa con run_all.sh). Si no, lo busca
    bajo /kaggle/input: así un notebook puede reutilizar la salida de otro notebook
    añadida como Input, sin repetir pasos costosos (p. ej. la traducción).
    """
    if os.path.exists(local_path):
        return local_path
    skip = {"train_series", "test_series", "train_images", "test_images"}
    if os.path.isdir(search_root):
        for root, dirs, files in os.walk(search_root):
            dirs[:] = [d for d in dirs if d not in skip]
            if name in files:
                path = os.path.join(root, name)
                print(f"AVISO: {local_path} no existe; se usa {path}")
                return path
    raise FileNotFoundError(f"No se encontró {name} ({local_path} ni bajo {search_root}). "
                            "Ejecuta el paso previo o añade como Input el notebook que lo generó.")


def find_dir(name, local_path, marker, search_root="/kaggle/input"):
    """Como find_file, pero para una carpeta que contiene el archivo `marker`."""
    if os.path.isfile(os.path.join(local_path, marker)):
        return local_path
    path = find_file(marker, os.path.join(local_path, marker), search_root)
    return os.path.dirname(path)

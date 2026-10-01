# P1 — Clasificación multietiqueta de anomalías de rodilla en MRI

Examen parcial · Redes Neuronales y Aprendizaje Profundo · UNI FIIS 2026-II

## Datos

RSNA Knee Abnormality Detection (Kaggle). Los datos **no** están en este
repositorio (reglas de la competencia). Se usan montados en un Kaggle Notebook.

## Ejecución (comando único)

En un Kaggle Notebook con la competencia añadida como Input
(**+ Add Input → RSNA Knee Abnormality Detection**), Internet activado y GPU T4:

```bash
!git clone https://github.com/<usuario>/Pregunta1_Parcial_RNAP.git
!cd Pregunta1_Parcial_RNAP && bash run_all.sh /kaggle/input/competitions/rsna-knee-abnormality-detection
```

Los resultados quedan en `Pregunta1_Parcial_RNAP/outputs/`.

## Estructura

```
run_all.sh        comando único: ejecuta todos los pasos en orden
requirements.txt  dependencias
scripts/          un script por paso del pipeline (01_eda.py, 02_..., ...)
src/knee/         código reutilizable (etiquetas, datasets, modelos, métricas)
outputs/          resultados generados (no se versionan)
```

## Pipeline

| Paso | Script | Qué hace |
|---|---|---|
| 1 | `scripts/01_eda.py` | Exploración de datos (solo estadísticas agregadas) |
| 2 | `scripts/02_translate_reports.py` | Aplica un glosario clínico por idioma (`src/knee/glossary.py`) y traduce al inglés con NLLB-200 (GPU, local en Kaggle) |
| 3 | `scripts/03_extract_labels.py` | Extrae las 12 etiquetas débiles con reglas y negación (`src/knee/extractor.py`) |
| 4 | `scripts/04_validate_labels.py` | Mide el acuerdo del extractor contra los 58 estudios gold |
| 5 | `scripts/05_build_subset.py` | Elige 600 estudios no gold estratificados por etiqueta; validación interna agrupada por escáner; gold = test |
| 6 | `scripts/06_preprocess.py` | DICOM -> caché `.npy` uint8 (16 cortes ordenados, 256×256, planos sagital y coronal) |

Los reportes nunca salen de Kaggle ni se envían a APIs externas: la traducción usa un
modelo abierto que corre en el propio notebook. `outputs/` no se versiona.

Pruebas: `python tests/test_extractor.py` y `python tests/test_glossary.py`

### Versiones del extractor

- **v1**: reglas en inglés + traducción NLLB sin glosario (etiqueta git `extractor-v1`).
- **v2**: glosario clínico antes de traducir + reglas ampliadas. Los cambios se derivaron
  solo de `tools/vocab_mining.py` sobre estudios no gold, sin usar etiquetas.

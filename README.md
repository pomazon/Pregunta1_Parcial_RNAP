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
| 2 | `scripts/02_translate_reports.py` | Traduce los reportes no ingleses al inglés con NLLB-200 (GPU, local en Kaggle) |
| 3 | `scripts/03_extract_labels.py` | Extrae las 12 etiquetas débiles con reglas y negación (`src/knee/extractor.py`) |
| 4 | `scripts/04_validate_labels.py` | Mide el acuerdo del extractor contra los 58 estudios gold |

Los reportes nunca salen de Kaggle ni se envían a APIs externas: la traducción usa un
modelo abierto que corre en el propio notebook. `outputs/` no se versiona.

Pruebas del extractor: `python tests/test_extractor.py`

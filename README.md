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
!cd Pregunta1_Parcial_RNAP && bash run_all.sh /kaggle/input/rsna-knee-abnormality-detection
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

# P1 — Clasificación multietiqueta de anomalías de rodilla en MRI

Examen parcial · Redes Neuronales y Aprendizaje Profundo · UNI FIIS 2026-II

Modelo 2.5D a nivel de estudio que predice 12 hallazgos en resonancias de rodilla, entrenado con
etiquetas débiles extraídas de reportes radiológicos multilingües y evaluado en los 58 estudios con
etiqueta dorada (gold).

## Datos

RSNA Knee Abnormality Detection (Kaggle). Los datos **no** están en este repositorio (reglas de la
competencia): se usan montados en un Kaggle Notebook. Ningún reporte, imagen ni etiqueta derivada se
versiona (`outputs/` está en `.gitignore`) y los reportes no se envían a ninguna API externa: la
traducción usa un modelo abierto (NLLB-200) que corre dentro del notebook.

## Ejecución (comando único)

**Entorno recomendado: Kaggle Notebook.** Es donde se desarrolló y verificó el pipeline, y tiene tres
ventajas prácticas:

- **Los datos ya están montados** en `/kaggle/input/…` al añadir la competencia como Input: no hay que
  descargar los miles de estudios DICOM ni tener espacio en disco para ellos.
- **GPU T4 gratuita**, necesaria para la traducción (NLLB-200) y el entrenamiento.
- **Los datos no salen de Kaggle**, tal como piden las reglas de la competencia.

También puede ejecutarse fuera de Kaggle (ver [Ejecución fuera de Kaggle](#ejecución-fuera-de-kaggle)),
pero exige descargar la competencia completa.

En un **Kaggle Notebook** nuevo:

1. **+ Add Input** → competencia *RSNA Knee Abnormality Detection*.
2. *Settings*: **Accelerator = GPU T4**, **Internet = On**.
3. Ejecutar:

```bash
!git clone https://github.com/pomazon/Pregunta1_Parcial_RNAP.git
!cd Pregunta1_Parcial_RNAP && bash run_all.sh
```

`run_all.sh` instala `requirements.txt` y ejecuta los pasos 1 a 10 en orden. Los resultados quedan en
`Pregunta1_Parcial_RNAP/outputs/`. Si la competencia está montada en otra ruta, pasarla como argumento:
`bash run_all.sh /ruta/a/los/datos`.

### Prueba rápida (recomendada para verificar)

```bash
!cd Pregunta1_Parcial_RNAP && bash run_all.sh --quick
```

Corre **todos** los pasos de extremo a extremo con pocos estudios (200 reportes traducidos incluyendo
los 58 gold, 80 estudios para entrenar, 20 por partición en imágenes) y **1 epoch** por experimento.
Sirve para comprobar que el pipeline funciona sin esperar las horas de la corrida completa; **sus
métricas no son resultados**. Usar un notebook limpio, porque escribe en `outputs/`.

Verificada en Kaggle (GPU T4) el 24-oct-2026: los 10 pasos terminaron sin errores en 25 min.

### Corrida completa

El mismo comando de arriba, sin `--quick`, reproduce los resultados del informe:

```bash
!cd Pregunta1_Parcial_RNAP && bash run_all.sh
```

En una GPU T4 de Kaggle toma varias horas; lo más costoso son la traducción y los seis
entrenamientos (el preprocesamiento tarda unos 4 min).

<details>
<summary>Si una sola sesión no alcanza (límite de 12 h de Kaggle)</summary>

La corrida se puede partir en tramos, cada uno en un notebook nuevo que añade como Input la salida del
anterior. Los scripts buscan bajo `/kaggle/input` lo que no encuentran en `outputs/`
(ver `find_file` en `src/knee/common.py`), así que no hay que mover archivos a mano. En cada
notebook, clonar el repositorio y ejecutar antes `pip install -q -r requirements.txt`.

| Tramo | Comandos | Input |
|---|---|---|
| 1 | `01` a `06` (exploración, traducción, etiquetas, validación, subconjunto, caché) | Competencia |
| 2 | `bash scripts/run_experiments.sh` (entrenamientos A–F y resumen) | Salida del tramo 1 |
| 3 | `python scripts/09_gradcam.py --run C_resnet50_focal_soft` y `python scripts/10_figures.py` | Salidas de los tramos 1 y 2 |

Así generamos nosotros los resultados del informe, repartidos en varios notebooks de Kaggle.
</details>

## Ejecución fuera de Kaggle

El código no depende de Kaggle: `run_all.sh` recibe la carpeta de datos como argumento. Requisitos:
GPU con CUDA, Bash (Linux, macOS, Colab; en Windows usar WSL), Internet para descargar NLLB-200 y los
pesos de ResNet-50, y **espacio en disco para la competencia completa** (ver el tamaño en la pestaña
*Data* de la competencia; en Colab gratuito puede no alcanzar). Este entorno no fue verificado por
nosotros; el verificado es Kaggle.

```bash
# 1) Aceptar las reglas de la competencia en kaggle.com con la propia cuenta.
# 2) Configurar la API de Kaggle (KAGGLE_USERNAME y KAGGLE_KEY, o ~/.kaggle/kaggle.json).
pip install kaggle
kaggle competitions download -c rsna-knee-abnormality-detection -p data
cd data && unzip -q *.zip && rm *.zip && cd ..
# La carpeta debe contener train.csv, train_series.csv y train_series/
git clone https://github.com/pomazon/Pregunta1_Parcial_RNAP.git
cd Pregunta1_Parcial_RNAP && bash run_all.sh --quick ../data   # prueba rápida
#                         bash run_all.sh ../data           # corrida completa
```

Los datos descargados no deben compartirse ni subirse a ningún repositorio (reglas de la competencia).

## Estructura

```
run_all.sh          comando único (--quick para la prueba rápida)
requirements.txt    dependencias
scripts/            un script por paso del pipeline (01 … 10) y run_experiments.sh
src/knee/           código propio: etiquetas, glosario, extractor, dataset, modelos, pérdidas, métricas
tools/              vocab_mining.py (diagnóstico sin etiquetas) y plot_architecture.py (diagrama)
tests/              pruebas del extractor y del glosario con frases sintéticas
outputs/            resultados generados (no se versionan)
```

## Pipeline

| Paso | Script | Qué hace |
|---|---|---|
| 1 | `scripts/01_eda.py` | Exploración de datos (solo estadísticas agregadas) |
| 2 | `scripts/02_translate_reports.py` | Glosario clínico por idioma (`src/knee/glossary.py`) y traducción al inglés con NLLB-200 (GPU, local) |
| 3 | `scripts/03_extract_labels.py` | 12 etiquetas débiles por reglas con negación e incertidumbre (`src/knee/extractor.py`) |
| 4 | `scripts/04_validate_labels.py` | Acuerdo del extractor contra los 58 estudios gold (global, por etiqueta y por idioma) |
| 5 | `scripts/05_build_subset.py` | 600 estudios no gold estratificados por etiqueta; validación agrupada por escáner; gold = test |
| 6 | `scripts/06_preprocess.py` | DICOM → caché `.npy` uint8 (16 cortes ordenados, campo de visión 180 mm, 256×256, sagital y coronal) |
| 7 | `scripts/07_train.py` | Modelo 2.5D (CNN propia o ResNet-50 de timm + atención por hallazgo); evalúa en val y en los gold |
| 8 | `scripts/08_summarize.py` | Tabla comparativa y comparaciones pareadas por bootstrap (`scripts/run_experiments.sh` lanza A–F) |
| 9 | `scripts/09_gradcam.py` | Grad-CAM propio y atención por hallazgo (dónde mira cada hallazgo) |
| 10 | `scripts/10_figures.py` | Figuras (PDF/PNG) y tablas LaTeX del informe a partir de los resultados |

### Experimentos

| Id | Modelo | Pérdida | Etiquetas | Propósito |
|---|---|---|---|---|
| A | CNN desde cero | BCE | blandas | E2: modelo desde cero |
| B | ResNet-50 (ImageNet) | BCE | blandas | E2: transfer learning (base) |
| C | ResNet-50 | focal | blandas | E4/E6: ablación de pérdida |
| D | ResNet-50 | BCE | duras (incierto = 0) | E4: manejo de etiquetas ruidosas |
| E | ResNet-50 sin `pos_weight` | BCE | blandas | E6: evaluación de la mitigación del desbalance |
| F | ResNet-50, plano coronal | focal | blandas | E4: ablación de plano |

### Resultados principales (test = 58 estudios gold)

| Exp. | AUC macro [IC 95 %] | F1 macro |
|---|---|---|
| A · CNN propia | 0.586 [0.522–0.648] | 0.481 |
| B · ResNet-50 (BCE) | 0.695 [0.637–0.745] | 0.532 |
| **C · ResNet-50 focal** | **0.724 [0.671–0.772]** | **0.553** |
| D · etiquetas duras | 0.695 [0.640–0.743] | 0.535 |
| E · sin `pos_weight` | 0.706 [0.659–0.750] | 0.565 |
| F · coronal (focal) | 0.719 [0.665–0.769] | 0.563 |

## Pruebas

```bash
python tests/test_extractor.py
python tests/test_glossary.py
```

## Versiones

- `extractor-v1`: reglas en inglés + traducción NLLB sin glosario.
- `extractor-v2`: glosario clínico antes de traducir + reglas ampliadas, derivadas solo de
  `tools/vocab_mining.py` sobre estudios no gold, sin usar etiquetas. Evaluada una vez contra gold y congelada.
- `entrenamiento-v1` (A–D) y `entrenamiento-v2` (E, F, comparaciones y Grad-CAM).

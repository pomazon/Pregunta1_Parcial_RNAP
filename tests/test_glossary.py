"""Pruebas del glosario con frases sintéticas (no son datos de la competencia)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from knee.glossary import apply_glossary

CASES = [
    ("es", "Rotura del ligamento cruzado anterior. Derrame articular moderado.",
     ["tear", "anterior cruciate ligament", "joint effusion"]),
    ("es", "Menisco interno con rotura del cuerno posterior.", ["medial meniscus", "tear"]),
    ("tr", "Ön çapraz bağda yırtık. Eklem içi sıvı artışı. İç menisküste dejenerasyon.",
     ["anterior cruciate ligament", "tear", "joint fluid", "medial meniscus"]),
    ("tr", "Efüzyon izlenmedi. Baker kisti mevcut.", ["effusion", "Baker cyst"]),
    ("hr", "Ruptura prednje križne sveze. Izljev u zglobu. Edem koštane srži.",
     ["tear", "anterior cruciate ligament", "effusion", "bone marrow edema"]),
    ("de", "Innenmeniskushinterhornriss. Kleiner Gelenkerguss. Knorpelschaden retropatellar.",
     ["medial meniscus", "posterior horn", "tear", "joint effusion", "cartilage", "damage"]),
    ("de", "VKB-Ruptur. Knochenmarködem im lateralen Femurkondylus.",
     ["anterior cruciate ligament", "tear", "bone marrow edema"]),
    ("el", "Ρήξη του πρόσθιου χιαστού συνδέσμου. Συλλογή υγρού. Κάταγμα.",
     ["tear", "anterior cruciate ligament", "fluid collection", "fracture"]),
    ("bg", "Руптура на предната кръстна връзка. Излив в ставата. Медиален менискус.",
     ["tear", "anterior cruciate ligament", "effusion", "medial meniscus"]),
    ("nl", "Meniscusscheur. Gewrichtsvocht. Voorste kruisband intact.",
     ["tear", "joint fluid", "anterior cruciate ligament"]),
    ("fr", "Épanchement articulaire. Rupture du ligament croisé antérieur.",
     ["effusion", "tear", "anterior cruciate ligament"]),
    ("en", "Joint effusion.", ["Joint effusion"]),
]
fails = 0
for lang, text, expected in CASES:
    out = apply_glossary(text, lang)
    missing = [e for e in expected if e.lower() not in out.lower()]
    if missing:
        fails += 1
        print(f"FALLA [{lang}]: {out!r} -> falta {missing}")
print(f"{len(CASES) - fails}/{len(CASES)} casos correctos")
sys.exit(1 if fails else 0)

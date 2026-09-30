"""Nombres de las 12 etiquetas objetivo, tal como aparecen en train.csv.

Verificar contra la salida de scripts/01_eda.py y corregir aquí si difieren:
es el único lugar del código donde se definen.
"""
LABELS = [
    "ACL",               # lesión de ligamento cruzado anterior
    "MCL",               # lesión de ligamento colateral medial
    "Medial Meniscus",   # daño de menisco medial
    "Lateral Meniscus",  # daño de menisco lateral
    "Medial OA",         # osteoartrosis compartimento medial
    "Lateral OA",        # osteoartrosis compartimento lateral
    "PF OA",             # osteoartrosis patelofemoral
    "Effusion",          # derrame articular
    "Synovitis",         # sinovitis
    "Baker's",           # quiste de Baker
    "Contusion",         # contusión ósea
    "Fracture",          # fractura
]

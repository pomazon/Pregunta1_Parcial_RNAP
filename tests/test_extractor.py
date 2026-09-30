"""Pruebas del extractor con frases sintéticas (no son datos de la competencia)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from knee.extractor import extract_report

CASES = [
    ("Complete tear of the anterior cruciate ligament.", {"ACL": "pos"}),
    ("The ACL is intact.", {"ACL": "neg"}),
    ("No evidence of ACL tear.", {"ACL": "neg"}),
    ("The ACL is torn and the MCL is intact.", {"ACL": "pos", "MCL": "neg"}),
    ("The cruciate and collateral ligaments are intact.", {"ACL": "neg", "MCL": "neg"}),
    ("Grade 2 sprain of the medial collateral ligament.", {"MCL": "pos"}),
    ("Tear of the posterior horn of the medial meniscus.", {"Medial Meniscus": "pos", "Lateral Meniscus": "nm"}),
    ("Medial meniscus intact; lateral meniscus shows a radial tear.", {"Medial Meniscus": "neg", "Lateral Meniscus": "pos"}),
    ("The medial and lateral menisci are intact.", {"Medial Meniscus": "neg", "Lateral Meniscus": "neg"}),
    ("Intrasubstance degeneration of the medial meniscus without tear.", {"Medial Meniscus": "unc"}),
    ("A tear of the medial meniscus cannot be excluded.", {"Medial Meniscus": "unc"}),
    ("Moderate joint effusion.", {"Effusion": "pos"}),
    ("Physiological amount of joint fluid.", {"Effusion": "unc"}),
    ("No joint effusion or Baker's cyst.", {"Effusion": "neg", "Baker's": "neg"}),
    ("Small Baker cyst.", {"Baker's": "pos"}),
    ("Bone marrow edema in the lateral femoral condyle consistent with contusion.", {"Contusion": "pos"}),
    ("No fracture.", {"Fracture": "neg"}),
    ("Non-displaced fracture of the lateral tibial plateau.", {"Fracture": "pos"}),
    ("Trabecular microfracture of the tibia.", {"Contusion": "pos", "Fracture": "nm"}),
    ("Chondromalacia of the patella grade 2.", {"PF OA": "pos"}),
    ("Osteophytes and cartilage thinning in the medial compartment.", {"Medial OA": "pos", "Lateral OA": "nm"}),
    ("Tricompartmental osteoarthritis.", {"Medial OA": "pos", "Lateral OA": "pos", "PF OA": "pos"}),
    ("Mild degenerative changes of the knee.", {"Medial OA": "unc", "PF OA": "unc"}),
    ("Synovial thickening consistent with synovitis.", {"Synovitis": "pos"}),
    ("No signs of synovitis.", {"Synovitis": "neg"}),
    ("Rupture of the inner ligament. Inner meniscus: tear of the posterior horn.", {"MCL": "pos", "Medial Meniscus": "pos"}),
    ("No meniscal tear.", {"Medial Meniscus": "neg", "Lateral Meniscus": "neg"}),
    ("Meniscal tear.", {"Medial Meniscus": "unc", "Lateral Meniscus": "unc"}),
    ("The MCL is intact. The ACL is torn.", {"ACL": "pos", "MCL": "neg"}),
    ("ACL reconstruction with intact graft.", {"ACL": "neg"}),
    ("Absent ACL fibres, empty notch sign.", {"ACL": "pos"}),
    ("Lateral meniscus: bucket-handle tear. Medial meniscus: normal.", {"Lateral Meniscus": "pos", "Medial Meniscus": "neg"}),
    ("No fracture, but bone contusion of the medial femoral condyle.", {"Fracture": "neg", "Contusion": "pos"}),
]

fails = 0
for text, expected in CASES:
    got = extract_report(text)
    bad = {k: (v, got[k]) for k, v in expected.items() if got[k] != v}
    if bad:
        fails += 1
        print("FALLA:", text, "->", bad)
print(f"{len(CASES) - fails}/{len(CASES)} casos correctos")
sys.exit(1 if fails else 0)

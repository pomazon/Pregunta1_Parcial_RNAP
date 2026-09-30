"""Extractor de etiquetas débiles basado en reglas, sobre reportes en inglés.

Para cada una de las 12 etiquetas asigna un estado por estudio:
    pos  hallazgo afirmado            -> etiqueta 1, confianza alta
    unc  hallazgo dudoso o leve       -> etiqueta 0 (dura) / 0.5 (blanda), confianza baja
    neg  hallazgo negado o estructura normal -> etiqueta 0, confianza alta
    nm   no mencionado                -> etiqueta 0, confianza media

Diseño:
  * El reporte se divide en oraciones (salto de línea, . ; ! ?).
  * Etiquetas "de estructura" (LCA, LCM, meniscos, artrosis por compartimento):
    se localiza la estructura y se buscan términos anormales/normales en su
    ámbito, que termina donde empieza la mención de otra estructura.
  * Etiquetas "de hallazgo" (derrame, sinovitis, Baker, contusión, fractura):
    el término mismo es el hallazgo.
  * Negación estilo NegEx: señal de negación en las ~8 palabras previas (cortando
    en "but", "however", ...) o en las ~6 siguientes ("is not seen", "excluded").
  * Incertidumbre: "possible", "cannot be excluded", "suspicious for", ... -> unc.

Las reglas se escribieron a partir de terminología radiológica estándar,
SIN mirar las etiquetas gold.

Versión 2 (30-sep-2026): cambios derivados SOLO de tools/vocab_mining.py sobre
estudios no gold (sin etiquetas): plurales en artrosis, orden de palabras
traducido ("meniscus medial", "crossed ligament"), "liquid"/"amount of fluid",
edema de Hoffa como sinovitis incierta (criterio MOAKS de Hoffa-synovitis),
cambios de señal ósea como contusión incierta, fracturas antiguas como inciertas,
"strain" como término débil y herencia de encabezados ("Medial meniscus:").
"""
import re

from .labels import LABELS

POS, UNC, NEG, NM = "pos", "unc", "neg", "nm"
STATUS_TO_HARD = {POS: 1, UNC: 0, NEG: 0, NM: 0}
STATUS_TO_SOFT = {POS: 1.0, UNC: 0.5, NEG: 0.0, NM: 0.0}
STATUS_TO_CONF = {POS: 1.0, NEG: 1.0, NM: 0.7, UNC: 0.3}
_RANK = {POS: 3, UNC: 2, NEG: 1, NM: 0}

I = re.IGNORECASE


def _rx(pattern):
    return re.compile(pattern, I)


# ---------------------------------------------------------------- cues
NEG_PRE = _rx(r"\b(?:no|not|without|negative for|free of|absence of|neither|nor|"
              r"rules? out|ruled out|excluded?|denies|resolved|resolution of)\b")
NEG_POST = _rx(r"\b(?:not (?:seen|identified|present|demonstrated|detected|evident|noted|observed|found)|"
               r"absent|(?:is|are|was|were|been) excluded|ruled out|not appreciated)\b")
HEDGE = _rx(r"\b(?:possible|possibly|may|might|could|suspect\w*|suspicious|questionable|"
            r"equivocal|indeterminate|borderline|cannot be (?:excluded|ruled out)|"
            r"can ?not be (?:excluded|ruled out)|not (?:be )?excluded|cannot rule out|"
            r"differential|query|doubtful|uncertain)\b|\?")
TERMINATOR = _rx(r"\b(?:but|however|although|though|except|whereas|while|apart from|aside from|"
                 r"other than|otherwise)\b")

# ---------------------------------------------------------------- terms
NORMAL = _rx(r"\b(?:intact|normal\w*|unremarkable|preserved|continuous|homogene\w*|"
             r"regular|physiolog\w*|within normal limits|no (?:abnormalit\w*|patholog\w*)|"
             r"adequate\w*|appropriate|taut|tense|well[- ]defined)\b")
HISTORY = _rx(r"\b(?:old|healed|healing|consolidated|previous|prior|chronic|remote|status post|s/p|"
              r"post-?operative|post-?surgical|history of|sequel\w*|fixation|osteosynthesis)\b")

LIG_STRONG = _rx(r"\b(?:tear\w*|torn|ruptur\w*|disrupt\w*|discontinu\w*|sprain\w*|injur\w*|"
                 r"avuls\w*|lax\w*|insufficien\w*|lesion\w*|not (?:clearly )?visuali[sz]\w*|"
                 r"non-?visuali[sz]\w*|empty notch|stump|remnant|graft (?:tear|failure|rupture)|"
                 r"fib(?:er|re)s? (?:are )?absent|absent (?:fib(?:er|re)s?|ligament))\b")
LIG_WEAK = _rx(r"\b(?:strain\w*|o?edema\w*|thicken\w*|increased (?:signal|T2)|high signal|hyperintens\w*|"
               r"mucoid|degenerat\w*|elongat\w*|wavy|irregular\w*|attenuat\w*)\b")

MEN_STRONG = _rx(r"\b(?:tear\w*|torn|ruptur\w*|macerat\w*|fragment\w*|flap|bucket[- ]handle|"
                 r"displac\w*|extru\w*|lesion\w*|damage\w*|destr\w*|blunt\w*|truncat\w*|"
                 r"resect\w*|meniscectom\w*|cleavage|grade (?:3|iii|3a|3b))\b")
MEN_WEAK = _rx(r"\b(?:degenerat\w*|intrasubstance|mucoid|grade (?:1|2|i|ii)\b|"
               r"signal (?:alteration|change)\w*|increased (?:intra\w* )?signal|hyperintens\w*|"
               r"irregular\w*|parameniscal cyst|meniscal cyst)")

OA_STRONG = _rx(r"\b(?:osteoarthrit\w*|osteoarthros\w*|arthros\w*|gonarthros\w*|"
                r"degenerative (?:change|disease|joint|arthropath)\w*|chondromalaci\w*|chondropath\w*|"
                r"chondral (?:defect|loss|thinning|lesion|damage|fissur\w*|ulcer\w*|wear)\w*|"
                r"cartilag\w* (?:loss|thinning|defect|damage|lesion|fissur\w*|wear|narrowing|degenerat\w*|"
                r"ulcer\w*|erosion)\w*|chondrosis|"
                r"(?:thinning|loss|defects?|damage|fissur\w*|wear|lesions?|erosions?) of (?:the )?(?:\w+ ){0,2}cartilage|"
                r"grade (?:ii|iii|iv|2|3|4) (?:\w+ ){0,2}chondr\w*|chondr\w* (?:\w+ ){0,2}grade (?:ii|iii|iv|2|3|4)|"
                r"osteophyt\w*|joint space narrowing|narrowing of the joint space|eburnation|denuded|"
                r"subchondral (?:cyst\w*|sclerosis))\b")
OA_WEAK = _rx(r"\b(?:cartilag\w* (?:irregular\w*|heterogene\w*|signal|changes?|alterations?)|"
              r"chondral (?:irregular\w*|changes?|alterations?)|"
              r"subchondral (?:o?edema|bone marrow))\b")

# Hallazgos
EFFUSION = _rx(r"\b(?:effusion\w*|hydrops|hydrarthros\w*|(?:lipo)?ha?emarthros\w*|"
               r"(?:joint|articular|intra-?articular|synovial) (?:fluid|liquid)|"
               r"(?:increased|excess\w*|abundant) (?:joint |synovial |intra-?articular )?(?:fluid|liquid)|"
               r"(?:fluid|liquid) (?:collection|accumulation)?\s?(?:in|within|inside) the (?:knee |joint |articular )?"
               r"(?:joint|articular|suprapatellar|recess|cavity)\w*|"
               r"(?:increased|excess\w*|abundant|moderate|large|significant|marked|considerable|small|minimal|mild|"
               r"slight|some|little) (?:amount|quantity|volume) of (?:\w+ ){0,2}(?:fluid|liquid))\b")
EFFUSION_MILD = _rx(r"\b(?:physiolog\w*|trace|minimal|minute|tiny|small|slight|little|mild)\b")
SYNOVITIS = _rx(r"\b(?:synovitis|synovial (?:thickening|proliferation|hypertroph\w*|enhancement|"
                r"inflammation|reaction)|thicken\w* (?:of the )?synovi\w*|villonodular|pvns)\b")
# Hoffa-synovitis (MOAKS): alteración de señal del cuerpo adiposo de Hoffa -> sinovitis incierta
SYNOVITIS_WEAK = _rx(r"\b(?:hoffa\w*(?:'s)? (?:\w+ ){0,3}(?:o?edema|signal|hyperintens\w*|inflam\w*)|"
                     r"(?:o?edema|signal (?:change|alteration|abnormalit)\w*|hyperintens\w*|inflam\w*) "
                     r"(?:\w+ ){0,4}hoffa\w*)")
BAKER = _rx(r"\b(?:baker\w*|popliteal cyst\w*|popliteal (?:bursa\w*|fluid collection)|"
            r"(?:gastrocnemi\w*[- ](?:semimembranos\w*)|semimembranos\w*[- ]gastrocnemi\w*) "
            r"(?:bursa\w*|cyst\w*)|"
            r"cyst\w* (?:\w+ ){0,4}(?:popliteal|semi-?membran\w*|gastrocnemi\w*)|"
            r"(?:popliteal|semi-?membran\w*|gastrocnemi\w*) (?:\w+ ){0,4}cyst\w*)")
CONTUSION = _rx(r"\b(?:(?:bone )?contusion\w*|bruis\w*|(?:bone )?marrow (?:o?edema|signal|hyperintens\w*|lesion)\w*|"
                r"o?edema (?:of|in|within) the (?:bone )?marrow|bone o?edema|osseous (?:contusion|o?edema)|"
                r"trabecular (?:micro-?fractur\w*|injur\w*|o?edema)|microfractur\w*)\b")
CONTUSION_WEAK = _rx(r"\bsubchondral\b")
CONTUSION_WEAK_TERMS = _rx(r"\b(?:(?:bone|osseous|trabecular) (?:\w+ ){0,2}(?:signal )?(?:changes?|alterations?|"
                           r"abnormalit\w*|hyperintens\w*)|o?edema-?like|"
                           r"(?:signal )?(?:changes?|alterations?|abnormalit\w*|hyperintens\w*) (?:of|in|within) "
                           r"the (?:\w+ )?(?:bone|marrow|osseous)\w*)")
FRACTURE = _rx(r"\b(?:fractur\w*|fx|segond|impaction (?:injury|fracture)|osteochondral (?:fracture|fragment)|"
               r"cortical (?:break|disruption|interruption)|infraction)\b")

# ---------------------------------------------------------------- estructuras
# (regex, etiquetas a las que pertenece la mención). "_other" solo delimita.
OA3 = ["Medial OA", "Lateral OA", "PF OA"]
STRUCTURES = [
    (_rx(r"\b(?:anterior cruciate\w*|ACL|LCA|VKB|ÖÇB|OCB)\b"), ["ACL"]),
    (_rx(r"\banterior (?:crossed|cross|cruciform) ligament\w*"), ["ACL"]),
    (_rx(r"\bcruciate ligaments\b|\bcruciates\b|\bcross(?:ed)? ligaments\b"), ["ACL"]),
    (_rx(r"\bposterior (?:crossed|cross|cruciform) ligament\w*"), ["_other"]),
    (_rx(r"\b(?:cruciate and collateral|collateral and cruciate) ligaments\b"), ["ACL", "MCL"]),
    (_rx(r"\b(?:posterior cruciate\w*|PCL|LCP|HKB)\b"), ["_other"]),
    (_rx(r"\b(?:medial collateral\w*|tibial collateral\w*|MCL|LCM|LLI|MKB|"
         r"internal (?:lateral|collateral) ligament\w*|inner (?:band|ligament)\w*|"
         r"medial (?:lateral )?ligament\w*)\b"), ["MCL"]),
    (_rx(r"\b(?:lateral collateral\w*|fibular collateral\w*|LCL|LLE|"
         r"external (?:lateral|collateral) ligament\w*|outer (?:band|ligament)\w*)\b"), ["_other"]),
    (_rx(r"\bcollateral ligaments\b"), ["MCL"]),
    (_rx(r"\b(?:medial patellofemoral ligament|MPFL|retinacul\w*|patellar tendon|quadriceps tendon|"
         r"popliteus\w*|iliotibial\w*|posterolateral corner|biceps\w*|pes anserin\w*)\b"), ["_other"]),
    (_rx(r"\b(?:(?:medial|inner|internal) menisc\w*|menisc\w* (?:medial\w*|internal|inner))\b"), ["Medial Meniscus"]),
    (_rx(r"\b(?:(?:lateral|outer|external) menisc\w*|menisc\w* (?:lateral\w*|external|outer))\b"), ["Lateral Meniscus"]),
    (_rx(r"\b(?:menisci|both menisc\w*|(?:medial|internal|inner) and (?:lateral|external|outer) menisc\w*|"
         r"(?:lateral|external|outer) and (?:medial|internal|inner) menisc\w*)\b"),
     ["Medial Meniscus", "Lateral Meniscus"]),
    # Menisco sin lado: un positivo se degrada a incierto ("_generic")
    (_rx(r"\b(?:meniscal|meniscus|menisc(?:o|i)\w*)\b"), ["Medial Meniscus", "Lateral Meniscus", "_generic"]),
    (_rx(r"\b(?:medial|internal|inner) (?:femoro-?tibial |tibio-?femoral )?(?:compartment|joint space|"
         r"femoral condyle|tibial plateau|condyle|plateau|hemiplateau|femorotibial|tibiofemoral)\w*\b"),
     ["Medial OA"]),
    (_rx(r"\b(?:lateral|external|outer) (?:femoro-?tibial |tibio-?femoral )?(?:compartment|joint space|"
         r"femoral condyle|tibial plateau|condyle|plateau|hemiplateau|femorotibial|tibiofemoral)\w*\b"),
     ["Lateral OA"]),
    (_rx(r"\b(?:patell\w*|trochle\w*|femoro-?patell\w*|retropatell\w*)\b"), ["PF OA"]),
    (_rx(r"\b(?:tri-?compartment\w*|three compartments|all (?:three )?compartments|pan-?compartment\w*)\b"),
     OA3),
]
STRUCT_TERMS = {
    "ACL": (LIG_STRONG, LIG_WEAK), "MCL": (LIG_STRONG, LIG_WEAK),
    "Medial Meniscus": (MEN_STRONG, MEN_WEAK), "Lateral Meniscus": (MEN_STRONG, MEN_WEAK),
    "Medial OA": (OA_STRONG, OA_WEAK), "Lateral OA": (OA_STRONG, OA_WEAK), "PF OA": (OA_STRONG, OA_WEAK),
}
# etiqueta -> (términos fuertes, modificadores leves -> unc, términos débiles -> como máximo unc)
FINDINGS = {
    "Effusion": (EFFUSION, EFFUSION_MILD, None), "Synovitis": (SYNOVITIS, None, SYNOVITIS_WEAK),
    "Baker's": (BAKER, None, None), "Contusion": (CONTUSION, CONTUSION_WEAK, CONTUSION_WEAK_TERMS),
    "Fracture": (FRACTURE, HISTORY, None),
}
HEADER = re.compile(r"^[^.;!?]{2,60}:\s*$")


# ---------------------------------------------------------------- helpers
def split_sentences(text):
    parts = []
    for line in str(text).split("\n"):
        parts += [s.strip() for s in re.split(r"(?<=[.;!?])\s+", line) if s.strip()]
    return parts


def _last_words(text, n):
    cut = list(TERMINATOR.finditer(text))
    if cut:
        text = text[cut[-1].end():]
    return " ".join(text.split()[-n:])


def _first_words(text, n):
    m = TERMINATOR.search(text)
    if m:
        text = text[:m.start()]
    return " ".join(text.split()[:n])


def _classify(before, term, after, mild_rx=None, window_ctx=""):
    """Estado de una mención concreta de un término anormal/hallazgo."""
    pre, post = _last_words(before, 8), _first_words(after, 8)
    if HEDGE.search(pre) or HEDGE.search(post):
        return UNC
    if NEG_PRE.search(pre) or NEG_POST.search(post):
        return NEG
    if mild_rx is not None and (mild_rx.search(_last_words(before, 5)) or mild_rx.search(term)):
        return UNC
    return POS


def _merge(a, b):
    return a if _RANK[a] >= _RANK[b] else b


def _mentions(sentence):
    """Menciones de estructuras sin solapamiento (gana la más larga)."""
    raw = []
    for rx, labels in STRUCTURES:
        for m in rx.finditer(sentence):
            raw.append((m.start(), m.end(), labels))
    raw.sort(key=lambda t: (-(t[1] - t[0]), t[0]))
    kept = []
    for s, e, labels in raw:
        if all(e <= ks or s >= ke for ks, ke, _ in kept):
            kept.append((s, e, labels))
    return sorted(kept)


def _scope_status(sentence, left, start, end, right, strong, weak):
    """Estado de una estructura dentro de su ámbito.

    Si la estructura es la primera mención de la oración (left == 0), se combinan
    el texto previo y el posterior. Si hay otra estructura antes, el texto previo
    le pertenece a esa, así que solo se usa si después no se encontró nada.
    """
    if left == 0:
        return _part_status(sentence, left, right, right, strong, weak)
    for lo, hi in ((start, right), (left, start)):
        st = _part_status(sentence, lo, hi, right, strong, weak, left)
        if st != NM:
            return st
    return NM


def _part_status(sentence, lo, hi, right, strong, weak, left=None):
    """Estado según los términos anormales/normales de sentence[lo:hi]."""
    left = lo if left is None else left
    seg = sentence[lo:hi]
    status = NM
    for rx, is_weak in ((strong, False), (weak, True)):
        for m in rx.finditer(seg):
            before = sentence[left:lo + m.start()]
            st = _classify(before, m.group(0), sentence[lo + m.end():right])
            if st == POS and is_weak:
                st = NEG if NORMAL.search(_last_words(before, 3)) else UNC
            status = _merge(status, st)
    if status == NM and NORMAL.search(seg):
        status = NEG
    return status


def extract_sentence(sentence):
    out = {lab: NM for lab in LABELS}
    # --- etiquetas de estructura
    mentions = _mentions(sentence)
    for i, (s, e, labels) in enumerate(mentions):
        for lab in labels:
            if lab == "_other":
                continue
            others = [(ms, me) for ms, me, ml in mentions if lab not in ml]
            left = max([me for ms, me in others if me <= s], default=0)
            right = min([ms for ms, me in others if ms >= e], default=len(sentence))
            if lab == "_generic":
                continue
            strong, weak = STRUCT_TERMS[lab]
            st = _scope_status(sentence, left, s, e, right, strong, weak)
            if st == POS and "_generic" in labels:
                st = UNC
            out[lab] = _merge(out[lab], st)
    # --- artrosis sin compartimento: incierta para los tres
    has_compartment = any(set(l) & set(OA3) for _, _, l in mentions)
    if not has_compartment:
        for m in OA_STRONG.finditer(sentence):
            st = _classify(sentence[:m.start()], m.group(0), sentence[m.end():])
            if st == POS:
                for lab in OA3:
                    out[lab] = _merge(out[lab], UNC)
            elif st == NEG:
                for lab in OA3:
                    out[lab] = _merge(out[lab], NEG)
    # --- etiquetas de hallazgo
    for lab, (rx, mild, weak) in FINDINGS.items():
        for m in rx.finditer(sentence):
            if lab == "Fracture" and re.match(r"micro", sentence[max(0, m.start() - 5):m.start()], I):
                continue
            st = _classify(sentence[:m.start()], m.group(0), sentence[m.end():], mild)
            out[lab] = _merge(out[lab], st)
        if weak is not None:
            for m in weak.finditer(sentence):
                if lab == "Synovitis" and re.search(r"superolateral|supero-?lateral", m.group(0), I):
                    continue  # edema superolateral de Hoffa: síndrome de fricción, no sinovitis
                st = _classify(sentence[:m.start()], m.group(0), sentence[m.end():])
                out[lab] = _merge(out[lab], UNC if st == POS else st)
    return out


def iter_sentences(text):
    """Oraciones del reporte; las que no nombran ninguna estructura heredan el
    encabezado previo terminado en ':' (p. ej. "Medial meniscus:" en su propia línea)."""
    header = None
    for line in str(text).split("\n"):
        stripped = line.strip()
        if not stripped:
            header = None
            continue
        if HEADER.match(stripped):
            header = stripped if _mentions(stripped) else None
            yield stripped
            continue
        for sent in split_sentences(stripped):
            yield f"{header} {sent}" if header and not _mentions(sent) else sent


def extract_report(text):
    """Diccionario etiqueta -> estado (pos/unc/neg/nm) para un reporte completo."""
    status = {lab: NM for lab in LABELS}
    for sent in iter_sentences(text):
        for lab, st in extract_sentence(sent).items():
            status[lab] = _merge(status[lab], st)
    return status

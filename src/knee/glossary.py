"""Glosario de términos clínicos clave, idioma original -> inglés.

Se aplica ANTES de traducir con NLLB: se sustituyen en el texto original los
términos que el traductor suele deformar (p. ej. "derrame" o "efüzyon"
traducidos como "detachment", "excretion" o "diesel"; "Gleitlager" o
"hrskavica" que quedan sin traducir). NLLB conserva las palabras en inglés
insertadas, así que el extractor en inglés las reconoce.

Origen: terminología radiológica estándar de cada idioma, escrita sin mirar las
etiquetas gold. Se usan raíces con \\w* para cubrir declinaciones y sufijos
(turco, croata, búlgaro, griego) y compuestos alemanes.
"""
import re

_F = re.IGNORECASE | re.UNICODE

# Orden: frases largas antes que términos sueltos.
_RAW = {
    "es": [
        (r"ligamento\w* cruzado\w* anterior\w*", "anterior cruciate ligament"),
        (r"ligamento\w* cruzado\w* posterior\w*", "posterior cruciate ligament"),
        (r"ligamento\w* cruzado\w*", "cruciate ligaments"),
        (r"ligamento\w* (?:colateral|lateral) (?:medial|interno)\w*", "medial collateral ligament"),
        (r"ligamento\w* (?:colateral|lateral) (?:lateral|externo)\w*", "lateral collateral ligament"),
        (r"menisco\w* (?:medial|interno)\w*", "medial meniscus"),
        (r"menisco\w* (?:lateral|externo)\w*", "lateral meniscus"),
        (r"derrame\w* articular\w*", "joint effusion"),
        (r"derrame\w*", "effusion"),
        (r"l[ií]quido articular", "joint fluid"),
        (r"hidrartros\w*", "hydrarthrosis"),
        (r"sinovitis", "synovitis"),
        (r"quiste\w* de baker", "Baker cyst"),
        (r"quiste\w* popl[ií]teo\w*", "popliteal cyst"),
        (r"edema [óo]seo", "bone marrow edema"),
        (r"edema de (?:la )?m[ée]dula [óo]sea", "bone marrow edema"),
        (r"contusi[óo]n\w* [óo]se\w*", "bone contusion"),
        (r"fractura\w*", "fracture"),
        (r"rotura\w*|ruptura\w*|desgarro\w*", "tear"),
        (r"condromalacia\w*", "chondromalacia"),
        (r"condropat[ií]a\w*", "chondropathy"),
        (r"gonartrosis|artrosis", "osteoarthritis"),
        (r"cart[ií]lago\w*", "cartilage"),
        (r"tensi[óo]n (?:adecuada|normal|conservada)", "normal tension"),
    ],
    "tr": [
        (r"ön çapraz bağ\w*", "anterior cruciate ligament"),
        (r"arka çapraz bağ\w*", "posterior cruciate ligament"),
        (r"çapraz bağ\w*", "cruciate ligaments"),
        (r"(?:medial|iç) (?:kollateral|yan) bağ\w*", "medial collateral ligament"),
        (r"(?:lateral|dış) (?:kollateral|yan) bağ\w*", "lateral collateral ligament"),
        (r"(?:medial|iç) menisk\w*", "medial meniscus"),
        (r"(?:lateral|dış) menisk\w*", "lateral meniscus"),
        (r"eklem (?:içi )?(?:sıvı|mayi)\w*", "joint fluid"),
        (r"efüzyon\w*|effüzyon\w*", "effusion"),
        (r"(?:sıvı|mayi)\w*", "fluid"),
        (r"sinovit\w*", "synovitis"),
        (r"baker kist\w*", "Baker cyst"),
        (r"popliteal kist\w*", "popliteal cyst"),
        (r"kemik iliği ödem\w*", "bone marrow edema"),
        (r"(?:kemik )?kontüzyon\w*", "bone contusion"),
        (r"fraktür\w*|kırı[kğ]\w*", "fracture"),
        (r"yırtı[kğ]\w*|yırtılma\w*|rüptür\w*", "tear"),
        (r"kondromalazi\w*", "chondromalacia"),
        (r"kıkırdak\w*", "cartilage"),
        (r"gonartroz\w*|osteoartrit\w*|osteoartroz\w*", "osteoarthritis"),
        (r"dejeneratif", "degenerative"),
    ],
    "hr": [
        (r"prednj\w* križn\w* (?:ligament|svez)\w*", "anterior cruciate ligament"),
        (r"stražnj\w* križn\w* (?:ligament|svez)\w*", "posterior cruciate ligament"),
        (r"križn\w* (?:ligament|svez)\w*", "cruciate ligaments"),
        (r"(?:medijal|unutarnj|unutrašnj)\w* (?:kolateraln|pobočn)\w* (?:ligament|svez)\w*", "medial collateral ligament"),
        (r"(?:lateral|vanjsk)\w* (?:kolateraln|pobočn)\w* (?:ligament|svez)\w*", "lateral collateral ligament"),
        (r"(?:medijal|unutarnj|unutrašnj)\w* menisk\w*", "medial meniscus"),
        (r"(?:lateral|vanjsk)\w* menisk\w*", "lateral meniscus"),
        (r"izljev\w*", "effusion"),
        (r"tekućin\w*|tekucin\w*", "fluid"),
        (r"sinovitis\w*", "synovitis"),
        (r"bakerov\w* cist\w*|baker\w* cist\w*", "Baker cyst"),
        (r"edem\w* (?:koštan\w* srži|kosti)", "bone marrow edema"),
        (r"(?:koštan\w* )?kontuzij\w*|nagnječenj\w*", "bone contusion"),
        (r"prijelom\w*|frakturu?\w*", "fracture"),
        (r"ruptur\w*|rascjep\w*|puknuć\w*", "tear"),
        (r"hondromalacij\w*", "chondromalacia"),
        (r"hrskavic\w*", "cartilage"),
        (r"gonartroz\w*|osteoartritis\w*|artroz\w*", "osteoarthritis"),
        (r"degenerativn\w*", "degenerative"),
    ],
    # Alemán y neerlandés forman compuestos (Innenmeniskushinterhornriss): se sustituyen
    # subcadenas dentro de la palabra (ver _SUBSTRING_LANGS) y se rodean de espacios.
    "de": [
        (r"vorder\w* kreuzband\w*|vkb", " anterior cruciate ligament "),
        (r"hinter\w* kreuzband\w*|hkb", " posterior cruciate ligament "),
        (r"kreuzbänder|kreuzbaender", " cruciate ligaments "),
        (r"innenband", " medial collateral ligament "),
        (r"außenband|aussenband", " lateral collateral ligament "),
        (r"innenmeniskus", " medial meniscus "),
        (r"außenmeniskus|aussenmeniskus", " lateral meniscus "),
        (r"hinterhorn", " posterior horn "),
        (r"vorderhorn", " anterior horn "),
        (r"gelenkergu(?:ss|ß)\w*", " joint effusion "),
        (r"ergu(?:ss|ß)\w*", " effusion "),
        (r"synovialitis|synovitis", " synovitis "),
        (r"baker[- ]?zyste\w*", " Baker cyst "),
        (r"knochenmarks?(?:ödem|oedem)\w*|knochenödem\w*|knochenoedem\w*", " bone marrow edema "),
        (r"knochenkontusion\w*|bone bruise", " bone contusion "),
        (r"fraktur\w*", " fracture "),
        (r"ri(?:ss|ß)\w*|ruptur\w*", " tear "),
        (r"chondromalazie\w*", " chondromalacia "),
        (r"chondropathie\w*", " chondropathy "),
        (r"gonarthrose\w*|arthrose\w*", " osteoarthritis "),
        (r"knorpel", " cartilage "),
        (r"schaden|schäden", " damage "),
        (r"defekt\w*", " defect "),
        (r"gleitlager\w*", " trochlea "),
        (r"hoffa[- ]?fettkörper\w*|fettkörper\w*", " Hoffa fat pad "),
    ],
    "el": [
        (r"πρόσθι\w* χιαστ\w*(?: σύνδεσμ\w*)?|προσθι\w* χιαστ\w*(?: συνδεσμ\w*)?", "anterior cruciate ligament"),
        (r"οπίσθι\w* χιαστ\w*(?: σύνδεσμ\w*)?|οπισθι\w* χιαστ\w*(?: συνδεσμ\w*)?", "posterior cruciate ligament"),
        (r"χιαστ\w* σύνδεσμ\w*|χιαστ\w* συνδεσμ\w*", "cruciate ligaments"),
        (r"έσω πλάγι\w* σύνδεσμ\w*|εσω πλαγι\w* συνδεσμ\w*", "medial collateral ligament"),
        (r"έξω πλάγι\w* σύνδεσμ\w*|εξω πλαγι\w* συνδεσμ\w*", "lateral collateral ligament"),
        (r"έσω μηνίσκ\w*|εσω μηνισκ\w*", "medial meniscus"),
        (r"έξω μηνίσκ\w*|εξω μηνισκ\w*", "lateral meniscus"),
        (r"υδάρθρωσ\w*|υδαρθρωσ\w*", "hydrarthrosis"),
        (r"συλλογ\w* υγρ\w*", "fluid collection in the joint"),
        (r"υγρ\w*", "fluid"),
        (r"υμενίτιδ\w*|υμενιτιδ\w*|θυλακίτιδ\w*", "synovitis"),
        (r"κύστ\w* (?:του )?baker|κυστ\w* (?:του )?baker|κύστ\w* baker", "Baker cyst"),
        (r"οίδημα μυελού|οιδημα μυελου|οστικ\w* οίδημα|οστικ\w* οιδημα|οίδημα του μυελού", "bone marrow edema"),
        (r"θλάσ\w*|θλασ\w*", "bone contusion"),
        (r"κάταγμ\w*|καταγμ\w*|κατάγματ\w*", "fracture"),
        (r"ρήξ\w*|ρηξ\w*", "tear"),
        (r"χονδρομαλακί\w*|χονδρομαλακι\w*", "chondromalacia"),
        (r"χόνδρ\w*|χονδρ\w*", "cartilage"),
        (r"οστεοαρθρ\w*|αρθρίτιδ\w*|αρθρωσ\w*", "osteoarthritis"),
        (r"εκφυλιστικ\w*", "degenerative"),
    ],
    "bg": [
        (r"предн\w* кръстн\w* връзк\w*", "anterior cruciate ligament"),
        (r"задн\w* кръстн\w* връзк\w*", "posterior cruciate ligament"),
        (r"кръстн\w* връзк\w*", "cruciate ligaments"),
        (r"(?:медиал|вътреш)\w* (?:колатерал|странич)\w* връзк\w*", "medial collateral ligament"),
        (r"(?:латерал|външ)\w* (?:колатерал|странич)\w* връзк\w*", "lateral collateral ligament"),
        (r"(?:медиал|вътреш)\w* менискус\w*|(?:медиал|вътреш)\w* мениск\w*", "medial meniscus"),
        (r"(?:латерал|външ)\w* менискус\w*|(?:латерал|външ)\w* мениск\w*", "lateral meniscus"),
        (r"излив\w*", "effusion"),
        (r"течност\w*", "fluid"),
        (r"синовит\w*", "synovitis"),
        (r"киста на бейкър|бейкерова киста|бейкър\w* киста", "Baker cyst"),
        (r"костномозъчен оток|костен мозъчен оток|оток на костния мозък|костен оток", "bone marrow edema"),
        (r"(?:костна )?контузи\w*", "bone contusion"),
        (r"фрактур\w*|счупван\w*", "fracture"),
        (r"руптур\w*|разкъсван\w*", "tear"),
        (r"хондромалаци\w*", "chondromalacia"),
        (r"хрущял\w*", "cartilage"),
        (r"гонартроз\w*|остеоартр\w*|артроз\w*", "osteoarthritis"),
        (r"дегенеративн\w*", "degenerative"),
    ],
    "nl": [
        (r"voorste kruisband\w*", " anterior cruciate ligament "),
        (r"achterste kruisband\w*", " posterior cruciate ligament "),
        (r"kruisbanden", " cruciate ligaments "),
        (r"mediale (?:collaterale )?band\w*", " medial collateral ligament "),
        (r"laterale (?:collaterale )?band\w*", " lateral collateral ligament "),
        (r"gewrichtsvocht", " joint fluid "),
        (r"gewrichtseffusie", " joint effusion "),
        (r"vocht", " fluid "),
        (r"baker[- ]?cyste\w*", " Baker cyst "),
        (r"beenmergoedeem|botoedeem|bone bruise", " bone marrow edema "),
        (r"fractuur|breuk", " fracture "),
        (r"scheur\w*", " tear "),
        (r"kraakbeen", " cartilage "),
        (r"gonartrose|artrose", " osteoarthritis "),
    ],
    "fr": [
        (r"ligament\w* crois[ée]\w* ant[ée]rieur\w*", "anterior cruciate ligament"),
        (r"ligament\w* crois[ée]\w* post[ée]rieur\w*", "posterior cruciate ligament"),
        (r"ligament\w* (?:collat[ée]ral|lat[ée]ral) (?:m[ée]dial|interne)\w*", "medial collateral ligament"),
        (r"ligament\w* (?:collat[ée]ral|lat[ée]ral) (?:lat[ée]ral|externe)\w*", "lateral collateral ligament"),
        (r"m[ée]nisque\w* (?:interne|m[ée]dial)\w*", "medial meniscus"),
        (r"m[ée]nisque\w* (?:externe|lat[ée]ral)\w*", "lateral meniscus"),
        (r"[ée]panchement\w*", "effusion"),
        (r"hydarthrose\w*", "hydrarthrosis"),
        (r"synovite\w*", "synovitis"),
        (r"kyste\w* (?:de baker|poplit[ée]\w*)", "Baker cyst"),
        (r"(?:œ|oe)d[èe]me osseux|contusion\w* osseuse\w*", "bone marrow edema"),
        (r"d[ée]chirure\w*|rupture\w*", "tear"),
        (r"chondropathie\w*", "chondropathy"),
        (r"gonarthrose\w*|arthrose\w*", "osteoarthritis"),
    ],
}

_SUBSTRING_LANGS = {"de", "nl"}


def _compile(lang, pattern):
    if lang in _SUBSTRING_LANGS:
        return re.compile(r"(?:" + pattern + r")", _F)
    return re.compile(r"(?<!\w)(?:" + pattern + r")(?!\w)", _F)


GLOSSARY = {lang: [(_compile(lang, p), en) for p, en in pairs] for lang, pairs in _RAW.items()}


def apply_glossary(text, lang):
    """Sustituye en `text` (idioma `lang`) los términos del glosario por su forma inglesa."""
    for rx, en in GLOSSARY.get(lang, []):
        text = rx.sub(en, text)
    if lang in _SUBSTRING_LANGS:
        text = re.sub(r"[ \t]{2,}", " ", text)
    return text

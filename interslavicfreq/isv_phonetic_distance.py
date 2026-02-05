# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import unicodedata
from typing import Dict, List, Tuple, Iterable, Optional, Union

Token = str
Seq = List[Token]
Alt = Tuple[Union[Token, Seq], float]  # (unit(s), weight)


# ------------------------------------------------------------
# 0) Weights / parameters (tune these)
# ------------------------------------------------------------

W = {
    "EXACT": 1.00,
    "VHIGH": 0.95,
    "HIGH": 0.90,
    "MID": 0.75,
    "LOW": 0.55,
    "VLOW": 0.35,
}

# Emission penalty biases "less typical" grapheme->ISV choices
EMIT_PENALTY = 0.15
DEFAULT_BEAM = 24

TRANSPOSITION_COST = 0.35  # for adjacent swap in Damerau-OSA


# ------------------------------------------------------------
# 1) ISV internal proximity (neighbors only)
# ------------------------------------------------------------

ISV_INTERNAL_NEAR: Dict[Token, List[Tuple[Token, float]]] = {
    # vowels
    "a":  [("å", W["HIGH"]), ("o", W["MID"]), ("ȯ", W["MID"]), ("ė", W["VLOW"])],
    "o":  [("å", W["HIGH"]), ("ų", W["MID"]), ("ȯ", W["MID"]), ("u", W["MID"])],
    "u":  [("ų", W["HIGH"]), ("o", W["MID"]), ("ȯ", W["MID"])],
    "i":  [("y", W["MID"]), ("ji", W["MID"])],
    "y":  [("i", W["MID"])],
    "e":  [("ė", W["HIGH"]), ("ě", W["MID"]), ("ȯ", W["VLOW"])],
    "ė":  [("e", W["HIGH"]), ("a", W["MID"]), ("ě", W["MID"])],
    "ě":  [("je", W["HIGH"]), ("e", W["MID"]), ("ė", W["MID"]), ("ę", W["MID"])],
    "å":  [("a", W["HIGH"]), ("o", W["HIGH"]), ("ȯ", W["MID"])],
    "ų":  [("u", W["HIGH"]), ("o", W["MID"])],
    "ę":  [("ja", W["MID"]), ("e", W["MID"]), ("ě", W["MID"])],

    # schwa/reduced vowel (NOT deletable)
    "ȯ":  [("a", W["MID"]), ("o", W["MID"]), ("u", W["MID"]),
           ("e", W["MID"]), ("i", W["MID"]), ("y", W["MID"]),
           ("å", W["MID"]), ("ė", W["MID"]), ("ě", W["MID"]),
           ("ų", W["MID"]), ("ę", W["MID"])],

    # iotation
    "ja": [("a", W["MID"]), ("ę", W["MID"]), ("j", W["LOW"])],
    "je": [("ě", W["HIGH"]), ("e", W["MID"]), ("j", W["LOW"])],
    "ji": [("i", W["HIGH"]), ("y", W["MID"]), ("j", W["LOW"])],
    "jo": [("o", W["MID"]), ("ȯ", W["MID"]), ("j", W["LOW"])],
    "ju": [("u", W["MID"]), ("ų", W["MID"]), ("j", W["LOW"])],

    # softness
    "t":  [("ť", W["HIGH"])],
    "d":  [("ď", W["HIGH"])],
    "n":  [("ń", W["HIGH"])],
    "l":  [("ľ", W["HIGH"])],
    "s":  [("ś", W["MID"]), ("š", W["VLOW"])],
    "z":  [("ź", W["MID"]), ("ž", W["VLOW"])],
    "š":  [("ž", W["LOW"]), ("s", W["VLOW"])],
    "ž":  [("š", W["LOW"]), ("z", W["VLOW"])],
    "c":  [("ć", W["MID"]), ("č", W["VLOW"]), ("dz", W["LOW"])],
    "č":  [("ć", W["MID"]), ("c", W["VLOW"]), ("dž", W["LOW"])],
    "dz": [("c", W["LOW"]), ("z", W["MID"])],
    "dž": [("č", W["LOW"]), ("ž", W["MID"]), ("đ", W["MID"])],

    # special đ: dž / žd-ish / soft-dž neighborhood
    "đ":  [("dž", W["MID"]), ("ď", W["LOW"]), ("ž", W["LOW"])],

    # clusters
    "šč": [("št", W["MID"]), ("š", W["MID"]), ("č", W["MID"])],
    "št": [("šč", W["MID"]), ("š", W["MID"]), ("t", W["MID"])],
}

# symmetric lookup
_NEAR: Dict[Tuple[Token, Token], float] = {}
for a, lst in ISV_INTERNAL_NEAR.items():
    for b, w in lst:
        _NEAR[(a, b)] = max(_NEAR.get((a, b), 0.0), w)
        _NEAR[(b, a)] = max(_NEAR.get((b, a), 0.0), w)

# weak voicing / close-but-not-zero pairs (as requested)
WEAK_PAIR_WEIGHT = W["LOW"]
WEAK_PAIRS = {
    ("k", "g"), ("p", "b"), ("t", "d"), ("s", "z"), ("f", "v"),
    ("š", "ž"), ("č", "dž"), ("c", "dz"),
    ("g", "h"), ("h", "g"),  # g~h very weak in many places (but still non-zero)
}
for a, b in list(WEAK_PAIRS):
    _NEAR.setdefault((a, b), WEAK_PAIR_WEIGHT)
    _NEAR.setdefault((b, a), WEAK_PAIR_WEIGHT)


ISV_VOWELS = {
    "a", "o", "u", "i", "y", "e", "ė", "ě", "å", "ų", "ę", "ȯ",
    "ja", "je", "ji", "jo", "ju",
}
# treat as "cheap" to insert/delete vs consonants
def ins_cost(tok: Token) -> float:
    if tok == "ȯ":
        return 0.45
    if tok in ISV_VOWELS:
        return 0.70
    if tok == "j":
        return 0.65
    return 1.00

def del_cost(tok: Token) -> float:
    return ins_cost(tok)

def sub_cost(a: Token, b: Token) -> float:
    if a == b:
        return 0.0
    w = _NEAR.get((a, b), 0.0)
    return 1.0 - w


# ------------------------------------------------------------
# 2) Token orders for greedy tokenization
# ------------------------------------------------------------

ISV_TOKEN_ORDER = [
    "šč", "št", "dž", "dz",
    "ja", "ju", "je", "ji", "jo",
    "ě", "ė", "ȯ", "å", "ų", "ę",
    "č", "š", "ž",
    "ć", "ľ", "ń", "ź", "ś", "ŕ", "ť", "ď", "đ",
]

TOKEN_ORDER = {
    "pl": ["szcz", "dź", "dż", "ch", "cz", "sz", "rz", "dz"],
    "cs": ["ch"],
    "sk": ["dž", "dz", "ch"],
    "hr": ["dž", "lj", "nj"],
    "sl": ["dž", "lj", "nj"],
    "sr_lat": ["dž", "lj", "nj"],
    "ru": ["жд", "дж", "дз", "шч"],
    "uk": ["дж", "дз"],
    "be": ["дж", "дз"],
    "bg": ["дж", "дз"],
    "mk": ["џ", "ѕ", "ѓ", "ќ", "љ", "њ"],
    "sr": ["џ", "љ", "њ", "ђ", "ћ"],
}


# ------------------------------------------------------------
# 3) Language->ISV mapping (only non-trivial cases; others fall back to identity)
# Supports multi-token expansions: unit can be a list like ["k","s"]
# ------------------------------------------------------------

LANG_TO_ISV: Dict[str, Dict[str, List[Alt]]] = {
    "pl": {
        "szcz": [("šč", W["HIGH"]), ("št", W["MID"])],
        "sz":   [("š", W["EXACT"])],
        "cz":   [("č", W["EXACT"])],
        "rz":   [("ž", W["VHIGH"]), ("r", W["LOW"]), ("ŕ", W["VLOW"])],
        "ch":   [("h", W["EXACT"])],
        "dz":   [("dz", W["EXACT"]), ("c", W["LOW"])],
        "dż":   [("dž", W["EXACT"]), ("đ", W["MID"])],
        "dź":   [("đ", W["VHIGH"]), ("ď", W["LOW"]), ("dž", W["MID"])],

        # 50/50 as requested (memory of old L-pronunciation)
        "ł":    [("v", W["HIGH"]), ("l", W["HIGH"]), ("u", W["MID"])],

        "w":    [("v", W["EXACT"]), ("f", W["LOW"])],
        "ó":    [("u", W["EXACT"]), ("ų", W["HIGH"]), ("o", W["MID"])],
        "ą":    [("ų", W["HIGH"]), ("o", W["MID"]), ("u", W["MID"])],
        "ę":    [("ę", W["HIGH"]), ("e", W["MID"]), ("ě", W["MID"])],

        "ć":    [("ć", W["EXACT"]), ("ť", W["MID"]), ("č", W["MID"])],
        "ś":    [("ś", W["EXACT"]), ("š", W["MID"]), ("s", W["MID"])],
        "ź":    [("ź", W["EXACT"]), ("ž", W["MID"]), ("z", W["MID"])],
        "ń":    [("ń", W["EXACT"]), ("n", W["MID"])],
        "ż":    [("ž", W["EXACT"]), ("š", W["LOW"])],
    },

    "cs": {
        "ch": [("h", W["EXACT"])],
        "ř":  [("r", W["HIGH"]), ("ž", W["MID"]), ("ŕ", W["MID"])],
        "ů":  [("u", W["EXACT"]), ("ų", W["MID"])],
        "ě":  [("ě", W["EXACT"]), ("je", W["HIGH"]), ("e", W["MID"])],
        "ď":  [("ď", W["EXACT"]), ("d", W["HIGH"]), ("đ", W["LOW"])],
        "ť":  [("ť", W["EXACT"]), ("t", W["HIGH"]), ("ć", W["LOW"])],
        "ň":  [("ń", W["EXACT"]), ("n", W["HIGH"])],
        "h":  [("h", W["HIGH"]), ("g", W["LOW"])],  # Czech /ɦ/
        # loan-ish:
        "x":  [(["k", "s"], W["MID"])],
        "q":  [(["k", "v"], W["MID"])],
        "w":  [("v", W["MID"])],
    },

    "sk": {
        "ch": [("h", W["EXACT"])],
        "dz": [("dz", W["EXACT"]), ("c", W["LOW"])],
        "dž": [("dž", W["EXACT"]), ("đ", W["MID"])],

        "ä":  [("ė", W["HIGH"]), ("e", W["MID"]), ("a", W["MID"])],
        "ô":  [("ų", W["HIGH"]), ("o", W["MID"]), ("u", W["MID"])],
        "ľ":  [("ľ", W["EXACT"]), ("l", W["HIGH"])],
        "ĺ":  [("l", W["EXACT"]), ("ľ", W["MID"])],
        "ŕ":  [("r", W["EXACT"]), ("ŕ", W["MID"])],
        "ň":  [("ń", W["EXACT"]), ("n", W["HIGH"])],
        "ď":  [("ď", W["EXACT"]), ("d", W["HIGH"])],
        "ť":  [("ť", W["EXACT"]), ("t", W["HIGH"])],
        "x":  [(["k", "s"], W["MID"])],
        "q":  [(["k", "v"], W["MID"])],
        "w":  [("v", W["MID"])],
    },

    "hr": {
        "dž": [("dž", W["EXACT"]), ("đ", W["MID"])],
        "lj": [("ľ", W["EXACT"]), ("l", W["HIGH"])],
        "nj": [("ń", W["EXACT"]), ("n", W["HIGH"])],
        "đ":  [("đ", W["EXACT"]), ("dž", W["MID"]), ("ď", W["LOW"])],
        "ć":  [("ć", W["EXACT"]), ("ť", W["MID"]), ("č", W["MID"])],
        "č":  [("č", W["EXACT"]), ("ć", W["MID"])],
        "š":  [("š", W["EXACT"])],
        "ž":  [("ž", W["EXACT"])],
    },

    "sl": {
        "dž": [("dž", W["EXACT"]), ("đ", W["MID"])],
        "lj": [("ľ", W["MID"]), ("l", W["MID"])],
        "nj": [("ń", W["MID"]), ("n", W["MID"])],
        "č":  [("č", W["EXACT"])],
        "š":  [("š", W["EXACT"])],
        "ž":  [("ž", W["EXACT"])],
    },

    # sr_lat: same as hr for our purposes
    "sr_lat": {
        "dž": [("dž", W["EXACT"]), ("đ", W["MID"])],
        "lj": [("ľ", W["EXACT"]), ("l", W["HIGH"])],
        "nj": [("ń", W["EXACT"]), ("n", W["HIGH"])],
        "đ":  [("đ", W["EXACT"]), ("dž", W["MID"]), ("ď", W["LOW"])],
        "ć":  [("ć", W["EXACT"]), ("ť", W["MID"]), ("č", W["MID"])],
        "č":  [("č", W["EXACT"]), ("ć", W["MID"])],
        "š":  [("š", W["EXACT"])],
        "ž":  [("ž", W["EXACT"])],
    },
}


# ------------------------------------------------------------
# 4) Normalization helpers
# ------------------------------------------------------------

APOSTROPHES = {"'", "’", "ʼ", "‘"}

def _normalize_text(text: str, keep_apostrophes: bool = False) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    out = []
    for ch in text:
        if ch.isalpha() or ch in {"č","š","ž","ě","ė","ȯ","å","ų","ę","ć","ľ","ń","ź","ś","ŕ","ť","ď","đ"}:
            out.append(ch)
        elif keep_apostrophes and ch in APOSTROPHES:
            out.append(ch)
        # else: drop (spaces/punct)
    return "".join(out)


# ------------------------------------------------------------
# 5) Tokenizers
# ------------------------------------------------------------

def _greedy_tokenize(text: str, patterns: List[str]) -> Seq:
    """Greedy longest-first tokenization using given patterns + fallback to single chars."""
    if not patterns:
        return list(text)
    # sort longest first
    pats = sorted(patterns, key=len, reverse=True)
    i = 0
    res: Seq = []
    n = len(text)
    while i < n:
        matched = False
        for p in pats:
            if text.startswith(p, i):
                res.append(p)
                i += len(p)
                matched = True
                break
        if not matched:
            res.append(text[i])
            i += 1
    return res


def _tokenize_isv(word: str) -> Seq:
    word = _normalize_text(word, keep_apostrophes=False)
    # greedy by ISV units
    i = 0
    res: Seq = []
    pats = sorted(ISV_TOKEN_ORDER, key=len, reverse=True)
    while i < len(word):
        matched = False
        for p in pats:
            if word.startswith(p, i):
                res.append(p)
                i += len(p)
                matched = True
                break
        if not matched:
            res.append(word[i])
            i += 1
    return res


_PL_VOWELS = set("aąeęioóuy")

def _tokenize_pl(word: str) -> Seq:
    """
    Polish:
    - greedy on szcz/sz/cz/rz/ch/dź/dż/dz
    - i-palatalization: ci/si/zi/ni/dzi + V -> ć/ś/ź/ń/đ + V (consume 'ci' etc)
    """
    word = _normalize_text(word, keep_apostrophes=False)
    i = 0
    res: Seq = []
    n = len(word)

    def peek(k: int) -> str:
        return word[i+k] if i+k < n else ""

    # patterns for greedy
    pats = TOKEN_ORDER["pl"]
    pats = sorted(pats, key=len, reverse=True)

    while i < n:
        # i-palatalization blocks first
        if word.startswith("dzi", i) and peek(3) in _PL_VOWELS:
            res.append("dź")  # represent as palatal affricate token; mapping will send to đ
            i += 3
            continue
        if word.startswith("ci", i) and peek(2) in _PL_VOWELS:
            res.append("ć")
            i += 2
            continue
        if word.startswith("si", i) and peek(2) in _PL_VOWELS:
            res.append("ś")
            i += 2
            continue
        if word.startswith("zi", i) and peek(2) in _PL_VOWELS:
            res.append("ź")
            i += 2
            continue
        if word.startswith("ni", i) and peek(2) in _PL_VOWELS:
            res.append("ń")
            i += 2
            continue

        # greedy digraphs/trigraphs
        matched = False
        for p in pats:
            if word.startswith(p, i):
                res.append(p)
                i += len(p)
                matched = True
                break
        if matched:
            continue

        res.append(word[i])
        i += 1

    return res


def _tokenize_latin_generic(word: str, lang: str) -> Seq:
    word = _normalize_text(word, keep_apostrophes=False)
    patterns = TOKEN_ORDER.get(lang, [])
    if lang in {"hr", "sl", "sr_lat"}:
        # these languages have lj/nj/dž frequently; keep them as units
        patterns = TOKEN_ORDER[lang]
    if lang in {"cs", "sk"}:
        patterns = TOKEN_ORDER[lang]
    return _greedy_tokenize(word, patterns)


# --- Cyrillic tokenization directly to ISV tokens (context iotation for ru/uk/be) ---

_CYR_BASE = {
    "а":"a","б":"b","в":"v","д":"d","ж":"ž","з":"z","й":"j","к":"k","л":"l","м":"m","н":"n",
    "о":"o","п":"p","р":"r","с":"s","т":"t","у":"u","ф":"f","х":"h","ц":"c","ч":"č","ш":"š",
}

def _tokenize_cyr(word: str, lang: str) -> Seq:
    """
    Returns ISV tokens directly (so mapping/beam is usually unnecessary for Cyrillic langs).
    Handles: ru/uk/be iotation (е ё ю я) via context; uk: є ї; be: ў; bg: ъ щ; sr/mk special letters.
    """
    # keep apostrophes for uk/be (as separators)
    keep_ap = lang in {"uk", "be"}
    s = _normalize_text(word, keep_apostrophes=keep_ap)
    if not s:
        return []

    # digraphs in Cyrillic inputs (ru/bg/uk/be)
    # We process them in-stream (longest first)
    digraphs = TOKEN_ORDER.get(lang, [])
    digraphs = sorted(digraphs, key=len, reverse=True)

    # iotation config
    if lang == "ru":
        iot = {"е": ("je", "e"), "ё": ("jo", "o"), "ю": ("ju", "u"), "я": ("ja", "a")}
        always = {}
        separators = {"ь", "ъ"}
        g_map = {"г": "g"}
        extra = {"щ": "šč", "ы": "y", "э": "e"}
        # ru "и" is /i/ (not ji in word start)
        i_letter = {"и": "i"}
    elif lang == "uk":
        iot = {"ю": ("ju", "u"), "я": ("ja", "a"), "е": ("je", "e")}  # 'е' exists too
        always = {"є": "je", "ї": "ji"}
        separators = {"ь"} | APOSTROPHES
        g_map = {"г": "h", "ґ": "g"}
        extra = {"щ": "šč", "и": "y", "і": "i"}
        i_letter = {"й": "j"}  # already in base
    elif lang == "be":
        iot = {"е": ("je", "e"), "ё": ("jo", "o"), "ю": ("ju", "u"), "я": ("ja", "a")}
        always = {}
        separators = {"ь"} | APOSTROPHES
        g_map = {"г": "h"}
        extra = {"ў": "v", "і": "i", "ы": "y", "э": "e", "щ": "šč"}
        i_letter = {}
    elif lang == "bg":
        iot = {}  # Bulgarian has no e/yo-type iotation like East Slavic
        always = {"я":"ja", "ю":"ju"}  # acceptable approximation for comparison
        separators = {"ь"}  # often silent; treat as separator
        g_map = {"г":"g"}
        extra = {"ъ":"ȯ", "щ":"št"}
        i_letter = {"и":"i"}  # bg и ~ i
    elif lang == "sr":
        iot = {}
        always = {}
        separators = set()
        g_map = {"г":"g"}
        extra = {"ј":"j","љ":"ľ","њ":"ń","ђ":"đ","ћ":"ć","џ":"dž"}
        i_letter = {"и":"i"}
    elif lang == "mk":
        iot = {}
        always = {}
        separators = set()
        g_map = {"г":"g"}
        extra = {"ј":"j","љ":"ľ","њ":"ń","ѓ":"đ","ќ":"ť","џ":"dž","ѕ":"dz"}
        i_letter = {"и":"i"}
    else:
        # fallback: treat as plain cyrillic -> base-ish
        iot, always, separators, g_map, extra, i_letter = {}, {}, set(), {}, {}, {}

    vowels_src = set("аеёиоуыэюяієїъ") | {"а","е","и","о","у","ы","э","ю","я","і","є","ї","ъ"}
    out: Seq = []

    force_j = False
    prev_is_vowel = False
    prev_is_boundary = True

    i = 0
    n = len(s)

    def emit(tok: Token):
        nonlocal prev_is_vowel, prev_is_boundary
        out.append(tok)
        prev_is_vowel = tok in ISV_VOWELS
        prev_is_boundary = False

    while i < n:
        # digraphs first (like дж, дз, жд, шч)
        matched = False
        for d in digraphs:
            if s.startswith(d, i):
                # map digraphs if known
                if lang in {"ru","uk","be","bg"}:
                    if d == "дж":
                        emit("dž")
                    elif d == "дз":
                        emit("dz")
                    elif d == "жд":
                        emit("đ")  # close bucket
                    elif d == "шч":
                        emit("šč")
                    else:
                        # unknown digraph -> split
                        pass
                elif lang == "mk":
                    # mk digraph list is single letters; ignore here
                    pass
                i += len(d)
                matched = True
                force_j = False
                break
        if matched:
            continue

        ch = s[i]

        # separators
        if ch in separators:
            force_j = True
            prev_is_vowel = False
            prev_is_boundary = False
            i += 1
            continue

        # boundary-like apostrophe already included in separators for uk/be

        # always-iotated letters
        if ch in always:
            emit(always[ch])
            force_j = False
            i += 1
            continue

        # iotation (context)
        if ch in iot:
            with_j, without_j = iot[ch]
            need_j = force_j or prev_is_boundary or prev_is_vowel
            emit(with_j if need_j else without_j)
            force_j = False
            i += 1
            continue

        # special extra mappings
        if ch in extra:
            emit(extra[ch])
            force_j = False
            i += 1
            continue

        # g/h map etc
        if ch in g_map:
            emit(g_map[ch])
            force_j = False
            i += 1
            continue

        # explicit i mapping for some languages
        if ch in i_letter:
            emit(i_letter[ch])
            force_j = False
            i += 1
            continue

        # base cyrillic
        if ch in _CYR_BASE:
            emit(_CYR_BASE[ch])
            force_j = False
            i += 1
            continue

        # unknown letter: keep (rare)
        emit(ch)
        force_j = False
        i += 1

    return out


# ------------------------------------------------------------
# 6) Beam expansion (multi-hypothesis mapping)
# ------------------------------------------------------------

def _as_units(u: Union[Token, Seq]) -> Seq:
    if isinstance(u, list):
        return u
    return [u]

def _expand_with_beam(tokens: Seq, mapping: Dict[str, List[Alt]], beam: int) -> List[Tuple[Seq, float]]:
    paths: List[Tuple[Seq, float]] = [([], 0.0)]
    for t in tokens:
        alts = mapping.get(t)
        if not alts:
            alts = [(t, 1.0)]
        new_paths: List[Tuple[Seq, float]] = []
        for seq, cost in paths:
            for unit, w in alts:
                units = _as_units(unit)
                new_seq = seq + units
                new_cost = cost + (1.0 - float(w)) * EMIT_PENALTY
                new_paths.append((new_seq, new_cost))
        new_paths.sort(key=lambda x: x[1])
        paths = new_paths[:beam]
    return paths


def _reduce_tokens(seq: Seq) -> Seq:
    """Simple metaphone-like compression: remove consecutive duplicates."""
    if not seq:
        return []
    out = [seq[0]]
    for t in seq[1:]:
        if t == out[-1]:
            continue
        out.append(t)
    return out


def _phonetic_hypotheses(word: str, lang: str, beam: int = DEFAULT_BEAM) -> List[Tuple[Seq, float]]:
    lang = lang.lower()

    if lang == "isv":
        seq = _reduce_tokens(_tokenize_isv(word))
        return [(seq, 0.0)]

    # Cyrillic languages -> directly to ISV tokens (usually single hypothesis)
    if lang in {"ru", "uk", "be", "bg", "sr", "mk"}:
        seq = _reduce_tokens(_tokenize_cyr(word, lang))
        return [(seq, 0.0)]

    # Latin-script languages
    if lang == "pl":
        toks = _tokenize_pl(word)
        paths = _expand_with_beam(toks, LANG_TO_ISV["pl"], beam)
        return [(_reduce_tokens(seq), c) for seq, c in paths]

    if lang in {"cs", "sk", "hr", "sl", "sr_lat"}:
        toks = _tokenize_latin_generic(word, lang)
        mapping = LANG_TO_ISV.get(lang, {})
        paths = _expand_with_beam(toks, mapping, beam)
        return [(_reduce_tokens(seq), c) for seq, c in paths]

    raise ValueError(f"Unsupported lang: {lang}")


# ------------------------------------------------------------
# 7) Weighted Damerau–Levenshtein (Optimal String Alignment)
# ------------------------------------------------------------

def _weighted_damerau_osa(a: Seq, b: Seq) -> float:
    n, m = len(a), len(b)
    if n == 0:
        return sum(ins_cost(x) for x in b)
    if m == 0:
        return sum(del_cost(x) for x in a)

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i-1][0] + del_cost(a[i-1])
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j-1] + ins_cost(b[j-1])

    for i in range(1, n + 1):
        ai = a[i-1]
        for j in range(1, m + 1):
            bj = b[j-1]

            dp[i][j] = min(
                dp[i-1][j] + del_cost(ai),              # deletion
                dp[i][j-1] + ins_cost(bj),              # insertion
                dp[i-1][j-1] + sub_cost(ai, bj),        # substitution
            )

            # transposition (adjacent swap)
            if i > 1 and j > 1 and a[i-1] == b[j-2] and a[i-2] == b[j-1]:
                dp[i][j] = min(dp[i][j], dp[i-2][j-2] + TRANSPOSITION_COST)

    return dp[n][m]


def _normalize_distance(raw_cost: float, len_a: int, len_b: int) -> float:
    denom = max(len_a, len_b, 1)
    # Costs are roughly <= denom (since ins/del/sub <=1), + small emission penalty later.
    d = raw_cost / float(denom)
    return max(0.0, min(1.0, d))


# ------------------------------------------------------------
# 8) Public API
# ------------------------------------------------------------

def phonetic_distance(isv_word: str, other_word: str, lang: str = "ru", beam: int = DEFAULT_BEAM) -> float:
    """
    Returns distance in [0..1]:
      0.0 = maximally close
      1.0 = far

    Example:
      isv_phonetic_distance("rěka", "rzeka", lang="pl")
    """
    isv_h = _phonetic_hypotheses(isv_word, "isv", beam=1)[0]
    isv_seq, isv_emit = isv_h  # emit is always 0 for ISV

    other_hyps = _phonetic_hypotheses(other_word, lang, beam=beam)

    best = 1.0
    for seq, emit_cost in other_hyps:
        raw = _weighted_damerau_osa(isv_seq, seq) + emit_cost + isv_emit
        d = _normalize_distance(raw, len(isv_seq), len(seq))
        if d < best:
            best = d

    return best


def phonetic_similarity(isv_word: str, other_word: str, lang: str = "ru", beam: int = DEFAULT_BEAM) -> float:
    """1 - distance"""
    return 1.0 - phonetic_distance(isv_word, other_word, lang=lang, beam=beam)


# ------------------------------------------------------------
# 9) Quick examples (optional)
# ------------------------------------------------------------
if __name__ == "__main__":
    pairs = [
        ("rěka", "река", "ru"),
        ("rěka", "rzeka", "pl"),
        ("svět", "свет", "ru"),
        ("noť", "noć", "hr"),
        ("ryba", "риба", "bg"),
        ("język", "język", "pl"),
        ("sȯn", "сън", "bg"),
    ]
    for isv, w, lg in pairs:
        print(lg, isv, w, "dist=", isv_phonetic_distance(isv, w, lang=lg), "sim=", isv_phonetic_similarity(isv, w, lang=lg))
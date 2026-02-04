import unicodedata

import regex

MARK_RE = regex.compile(r"[\p{Mn}\N{ARABIC TATWEEL}]", regex.V1)


def preprocess_text(text: str, language: str = "isv") -> str:
    """
    Preprocess text for Slavic languages.
    
    Simplified version:
    - NFC normalization
    - Case folding
    """
    # NFC normalization (подходит для латиницы и кириллицы)
    text = unicodedata.normalize("NFC", text)
    
    # Case folding
    text = text.casefold()

    return text


def remove_marks(text: str) -> str:
    """
    Remove decorations from words in abjad scripts:
    - Combining marks of class Mn, which tend to represent non-essential
      vowel markings.
    - Tatweels, horizontal segments that are used to extend or justify an
      Arabic word.
    """
    return MARK_RE.sub("", text)

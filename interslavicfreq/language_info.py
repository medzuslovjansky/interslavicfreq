from __future__ import annotations

def get_language_info(language: str) -> dict:
    """
    Simplified language info for Slavic languages.
    
    Returns basic settings suitable for Latin and Cyrillic scripts.
    This is a simplified version of the full wordfreq interface.
    """
    # Determine script based on language
    cyrillic_langs = {"ru", "uk", "be", "bg", "mk", "sr"}
    
    if language in cyrillic_langs:
        script = "Cyrl"
    else:
        script = "Latn"
    
    return {
        "tokenizer": "regex",
        "script": script,
        "remove_marks": False,
        "normal_form": "NFC",
        "diacritics_under": None,
        "transliteration": None,
        "lookup_transliteration": None,
    }
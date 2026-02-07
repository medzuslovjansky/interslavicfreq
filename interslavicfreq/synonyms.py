#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISV Synonym Builder & API (library module).

When run as __main__, downloads the word list, builds synonyms, and
saves a pickle cache next to the data files.

When imported, provides ``load_synonyms()`` which returns the synonym map
with transparent pickle caching.
"""
from __future__ import annotations

import os
import re
import pickle
import csv
import pandas as pd
from collections import defaultdict

from pathlib import Path
from .util import data_path as _data_path

from .transliteration_isv import transliterate_to_standard_latin


# ========================== CONFIG ==========================

CSV_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vRsEDDBEt3VXESqAgoQLUYHvsA5yMyujzGViXiamY7-yYrcORhrkEl5g6JZPorvJrgMk6sjUlFNT4Km"
    "/pub?output=csv"
)

LANG_COLUMNS = "en ru be uk pl cs sk bg mk sr hr sl".split()

# --- Настройки качества кластеризации ---
MIN_SHARED_LANGS = 7   # мин. языков с общим переводом (↑ строже, ↓ мягче)
MAX_DF           = 50   # переводы, встречающиеся чаще, — игнорируются

# ========================= HELPERS ==========================

_brackets1 = re.compile(r"\(.*?\)")
_brackets2 = re.compile(r"\[.*?\]")


def clear_cell(s):
    """Чистка: убрать # и ! в начале, скобки, привести к lower."""
    s = str(s).strip()
    s = s.lstrip("#!")
    s = _brackets1.sub("", s)
    s = _brackets2.sub("", s)
    return s.lower().strip()


def split_by_coma(text):
    """Разбить ячейку на варианты перевода."""
    text = str(text).strip()
    if not text:
        return []
    if ";" in text:
        return [x.strip() for x in text.split(";") if x.strip()]
    return [x.strip() for x in text.split(",") if x.strip()]


def normalize_pos(s):
    """Грубая нормализация части речи для группировки."""
    s = clear_cell(s).replace(". ", ".").replace("/", "")
    if s.startswith(("m.", "f.", "n.")):
        return "noun"
    return s.rstrip(".")


# ========================== LOAD ============================

def _get_synonyms_dir() -> Path:
    """Directory where synonym caches are stored (same level as data/)."""
    return _data_path().parent / "synonyms"


def _get_csv_cache_path() -> Path:
    d = _get_synonyms_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / "words_cache.csv"


def _get_pkl_path() -> Path:
    d = _get_synonyms_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / "synonyms.pkl"


def load_words(quiet: bool = True):
    csv_cache = _get_csv_cache_path()
    CSV_CACHE = str(csv_cache)
    if CSV_CACHE and os.path.isfile(CSV_CACHE):
        if not quiet:
            print(f"Читаем кэш: {CSV_CACHE}")
        df = pd.read_csv(CSV_CACHE, dtype=str, keep_default_na=False)
    else:
        if not quiet:
            print("Скачиваем CSV…")
        df = pd.read_csv(CSV_URL, dtype=str, keep_default_na=False)
        if CSV_CACHE:
            df.to_csv(CSV_CACHE, index=False)
            if not quiet:
                print(f"  CSV кэш сохранён: {CSV_CACHE}")

    needed = ["id", "isv", "partOfSpeech"] + LANG_COLUMNS
    df = df[[c for c in needed if c in df.columns]].fillna("")
    for c in df.columns:
        df[c] = df[c].apply(clear_cell)
    df["isv"] = df["isv"].apply(lambda x: transliterate_to_standard_latin(x))
    df["pos_norm"] = df["partOfSpeech"].apply(normalize_pos)
    return df


# ========================== BUILD ===========================

def build_synonyms(df, quiet: bool = True):
    n = len(df)
    available_langs = [l for l in LANG_COLUMNS if l in df.columns]

    # Превращаем в списки — быстрее, чем df.at в цикле
    pos_list = df["pos_norm"].tolist()
    isv_list = df["isv"].tolist()
    lang_lists = {lang: df[lang].tolist() for lang in available_langs}

    # 1) Инвертированный индекс: (lang, word) → {row_ids}
    inv = defaultdict(set)
    for i in range(n):
        for lang in available_langs:
            for w in split_by_coma(lang_lists[lang][i]):
                if w:
                    inv[(lang, w)].add(i)

    # Убираем стоп-слова (слишком частые переводы) и одиночки
    inv = {k: v for k, v in inv.items() if 1 < len(v) <= MAX_DF}
    if not quiet:
        print(f"  якорей после фильтрации: {len(inv)}")

    # 2) Для каждой строки находим синонимы
    synonyms_map = {}   # isv_variant → set(isv_synonyms)
    syn_col = [""] * n

    for i in range(n):
        my_pos = pos_list[i]
        my_isv_parts = [w for w in split_by_coma(isv_list[i]) if w]

        # Кандидаты: row_j → множество языков с совпадением
        cands = defaultdict(set)
        for lang in available_langs:
            for w in split_by_coma(lang_lists[lang][i]):
                if not w:
                    continue
                key = (lang, w)
                if key not in inv:
                    continue
                for j in inv[key]:
                    if j != i and pos_list[j] == my_pos:
                        cands[j].add(lang)

        # Собираем ISV-слова из строк, прошедших порог
        syns = set(my_isv_parts)
        for j, langs in cands.items():
            if len(langs) >= MIN_SHARED_LANGS:
                for w in split_by_coma(isv_list[j]):
                    if w:
                        syns.add(w)

        # Колонка для Excel (без самого себя)
        others = sorted(syns - set(my_isv_parts))
        syn_col[i] = ", ".join(others)

        # Маппинг для API
        for v in my_isv_parts:
            if v in synonyms_map:
                synonyms_map[v] |= syns
            else:
                synonyms_map[v] = set(syns)

    df["synonyms"] = syn_col
    return df, synonyms_map


# ========================== SAVE ============================

def save_all(df, syn_map, quiet: bool = True):
    pkl_path = _get_pkl_path()
    syn_dir = _get_synonyms_dir()

    # Pickle
    with open(pkl_path, "wb") as f:
        pickle.dump(syn_map, f, protocol=pickle.HIGHEST_PROTOCOL)
    if not quiet:
        print(f"  pickle: {pkl_path}")

    # Excel
    excel_path = syn_dir / "slovnik_synonyms.xlsx"
    try:
        df.to_excel(str(excel_path), index=False, engine="openpyxl")
        if not quiet:
            print(f"  excel:  {excel_path}")
    except Exception:
        pass  # openpyxl may not be installed

    # TSV
    tsv_path = syn_dir / "synonyms.tsv"
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["isv", "pos", "synonyms"])
        for word in sorted(syn_map):
            others = sorted(syn_map[word] - {word})
            if others:
                w.writerow([word, "", ", ".join(others)])
    if not quiet:
        print(f"  tsv:    {tsv_path}")


# ============================ API ===========================

def load_synonyms(use_cache: bool = True, quiet: bool = True) -> dict[str, set[str]]:
    """
    Load the synonym map.

    If ``use_cache`` is True and a pickle cache exists, it is loaded directly.
    Otherwise the CSV is downloaded (or read from local cache), synonyms are
    rebuilt, and the pickle is saved for next time.

    Returns:
        dict mapping ISV word → set of synonyms (including itself).
    """
    pkl_path = _get_pkl_path()

    if use_cache and pkl_path.exists():
        try:
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass  # corrupted cache → rebuild

    # Build from scratch
    df = load_words(quiet=quiet)
    if not quiet:
        print(f"Загружено строк: {len(df)}")
        print("Строим синонимы…")
    df, syn_map = build_synonyms(df, quiet=quiet)
    save_all(df, syn_map, quiet=quiet)
    return syn_map


def get_synonyms_raw(word: str, syn_map: dict[str, set[str]]) -> set[str]:
    """Low-level lookup: return synonyms from a pre-loaded map."""
    w = clear_cell(word)
    result = set(syn_map.get(w, set()))
    result.add(w)

    return result


# ========================== MAIN ============================
if __name__ == "__main__":
    df = load_words(quiet=False)
    df = load_words()
    print(f"Загружено строк: {len(df)}")

    df, syn_map = build_synonyms(df, quiet=False)
    df, syn_map = build_synonyms(df)

    n_with = sum(1 for v in syn_map.values() if len(v) > 1)
    print(f"Слов с синонимами: {n_with}")

    save_all(df, syn_map, quiet=False)
    save_all(df, syn_map)

    # Демо
    print("\n--- DEMO ---")
    demo = "mysliti dumati dom jezyk slovo govoriti ljubiti råbota voda noč dělati pisati".split()
    s = get_synonyms_raw(w, syn_map)
    s = synonyms(w)
    tag = f'{len(s)} шт.' if len(s) > 1 else "—"
    print(f'  synonyms("{w}") [{tag}] => {s}')

    # Крупнейшие синсеты
    print("\n--- Крупнейшие группы ---")
    seen = set()
    by_size = sorted(syn_map.items(), key=lambda x: -len(x[1]))
    for word, syns in by_size[:30]:
        key = frozenset(syns)
        if key in seen or len(syns) < 3:
            continue
        seen.add(key)
        print(f"  ({len(syns)}) {', '.join(sorted(syns))}")
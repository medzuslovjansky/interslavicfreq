# spellcheck_hunspell.py
from __future__ import annotations

import re
from pathlib import Path
from collections import defaultdict
from typing import Set, Dict, List, Optional
import pickle


class HunspellDictionary:
    """
    Упрощённый парсер hunspell словаря.
    Парсит .dic и .aff файлы, генерирует все формы слов.
    """

    def __init__(self, dic_path: Path, aff_path: Path, *, quiet: bool = False, 
                 lang: Optional[str] = None):
        self.words: Set[str] = set()
        self.prefixes: Dict[str, List[dict]] = defaultdict(list)
        self.suffixes: Dict[str, List[dict]] = defaultdict(list)
        self.encoding = "utf-8"
        self.flag_type = "single"  # single, long, num, UTF-8

        # Определяем язык из имени файла если не указан
        self.lang = lang or dic_path.stem

        if not quiet:
            print(f"  Парсинг {aff_path.name}...")
        self._parse_aff(aff_path)
        if not quiet:
            print(f"    Префиксов: {sum(len(v) for v in self.prefixes.values())}")
            print(f"    Суффиксов: {sum(len(v) for v in self.suffixes.values())}")

        if not quiet:
            print(f"  Парсинг {dic_path.name}...")
        self._parse_dic(dic_path)
        if not quiet:
            print(f"    Всего форм в словаре: {len(self.words)}")
        
        # Применяем фильтрацию для isv
        if self.lang == "isv":
            self._filter_isv_words()

    def _parse_aff(self, path: Path):
        """Парсит .aff файл."""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        match = re.search(r"^SET\s+(\S+)", content, re.MULTILINE)
        if match:
            self.encoding = match.group(1)

        match = re.search(r"^FLAG\s+(\S+)", content, re.MULTILINE)
        if match:
            self.flag_type = match.group(1).lower()

        try:
            with open(path, "r", encoding=self.encoding, errors="replace") as f:
                lines = f.readlines()
        except LookupError:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1

            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            if parts[0] in ("PFX", "SFX") and len(parts) == 4:
                try:
                    affix_type = parts[0]
                    flag = parts[1]
                    cross_product = parts[2] == "Y"
                    count = int(parts[3])

                    for _ in range(count):
                        if i >= len(lines):
                            break
                        rule_line = lines[i].strip()
                        i += 1

                        if not rule_line or rule_line.startswith("#"):
                            continue

                        rule_parts = rule_line.split()
                        if len(rule_parts) < 4:
                            continue

                        if rule_parts[0] == affix_type and rule_parts[1] == flag:
                            strip = rule_parts[2] if rule_parts[2] != "0" else ""

                            add_part = rule_parts[3]
                            if "/" in add_part:
                                add = add_part.split("/")[0]
                            else:
                                add = add_part if add_part != "0" else ""

                            condition = rule_parts[4] if len(rule_parts) > 4 else "."
                            if condition.startswith("po:") or condition.startswith("ds:"):
                                condition = "."

                            rule = {
                                "strip": strip,
                                "add": add,
                                "condition": condition,
                                "cross": cross_product,
                            }

                            if affix_type == "PFX":
                                self.prefixes[flag].append(rule)
                            else:
                                self.suffixes[flag].append(rule)
                except (ValueError, IndexError):
                    continue

    def _parse_flags(self, flag_str: str) -> List[str]:
        """Парсит строку флагов в зависимости от типа."""
        if not flag_str:
            return []

        if self.flag_type == "long":
            return [flag_str[i : i + 2] for i in range(0, len(flag_str), 2)]
        if self.flag_type == "num":
            return flag_str.split(",")
        return list(flag_str)

    def _parse_dic(self, path: Path):
        """Парсит .dic файл и генерирует все формы."""
        try:
            with open(path, "r", encoding=self.encoding, errors="replace") as f:
                lines = f.readlines()
        except LookupError:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

        start_line = 0
        if lines and lines[0].strip().isdigit():
            start_line = 1

        for line in lines[start_line:]:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("\t"):
                continue

            if "\t" in line:
                line = line.split("\t")[0]

            if "/" in line:
                parts = line.split("/", 1)
                word = parts[0].strip()
                flag_str = parts[1].split()[0] if " " in parts[1] else parts[1]
            else:
                word = line.split()[0] if " " in line else line
                flag_str = ""

            if not word:
                continue

            word_lower = word.lower()
            self.words.add(word_lower)
            self.words.add(word)  # оригинальный регистр тоже держим

            flags = self._parse_flags(flag_str)
            self._generate_forms(word_lower, flags)

    def _match_condition(self, word: str, condition: str, is_suffix: bool) -> bool:
        if condition == "." or not condition:
            return True
        try:
            pattern = (condition + "$") if is_suffix else ("^" + condition)
            return bool(re.search(pattern, word, re.IGNORECASE))
        except re.error:
            return True

    def _apply_suffix(self, word: str, rule: dict) -> Optional[str]:
        if not self._match_condition(word, rule["condition"], is_suffix=True):
            return None
        strip = rule["strip"]
        add = rule["add"]
        if strip:
            if word.endswith(strip):
                return word[: -len(strip)] + add
            return None
        return word + add

    def _apply_prefix(self, word: str, rule: dict) -> Optional[str]:
        if not self._match_condition(word, rule["condition"], is_suffix=False):
            return None
        strip = rule["strip"]
        add = rule["add"]
        if strip:
            if word.startswith(strip):
                return add + word[len(strip) :]
            return None
        return add + word

    def _generate_forms(self, word: str, flags: List[str]):
        forms = {word}

        suffix_forms = {word}
        for flag in flags:
            if flag in self.suffixes:
                for rule in self.suffixes[flag]:
                    new_form = self._apply_suffix(word, rule)
                    if new_form:
                        suffix_forms.add(new_form)
                        forms.add(new_form)

        for flag in flags:
            if flag in self.prefixes:
                for rule in self.prefixes[flag]:
                    new_form = self._apply_prefix(word, rule)
                    if new_form:
                        forms.add(new_form)

                    if rule["cross"]:
                        for suf_form in suffix_forms:
                            if suf_form != word:
                                new_form = self._apply_prefix(suf_form, rule)
                                if new_form:
                                    forms.add(new_form)

        self.words.update(forms)

    def check(self, word: str) -> bool:
        w = word.lower()
        return w in self.words or word in self.words

    # def _filter_isv_words(self):
    #     """
    #     Фильтрация для межславянского языка.
    #     Удаляем слова, заканчивающиеся на 'vš' или 'č' 
    #     (это некорректные формы).
    #     """
    #     self.words = {
    #         word for word in self.words 
    #         if not (word.endswith('vš') or word.endswith('č'))
    #     }

    #     return w in self.words or word in self.words

    def _filter_isv_words(self):
        """
        Фильтрация для межславянского языка.
        Удаляем слова, заканчивающиеся на 'vš' или 'č' 
        (это некорректные формы).
        """
        self.words = {
            word for word in self.words 
            if not (word.endswith('vš') or word.endswith('č'))
        }


def _get_cache_path(dic_path: Path) -> Path:
    """Получить путь к файлу кэша для словаря."""
    return dic_path.with_suffix(".cache.pickle")

def load_hunspell(dic_path: str | Path, aff_path: str | Path, *, 
                  quiet: bool = False, use_cache: bool = True) -> HunspellDictionary:
    """
    Загрузить Hunspell словарь.
    
    Args:
        dic_path: Путь к .dic файлу
        aff_path: Путь к .aff файлу
        quiet: Не выводить сообщения о прогрессе
        use_cache: Использовать кэширование в pickle
    """
    dic_path = Path(dic_path)
    aff_path = Path(aff_path)
    
    if not dic_path.exists():
        raise FileNotFoundError(f"Not found: {dic_path}")
    if not aff_path.exists():
        raise FileNotFoundError(f"Not found: {aff_path}")
    
    cache_path = _get_cache_path(dic_path)
    
    # Пробуем загрузить из кэша
    if use_cache and cache_path.exists():
        cache_mtime = cache_path.stat().st_mtime
        dic_mtime = dic_path.stat().st_mtime
        aff_mtime = aff_path.stat().st_mtime
        
        if cache_mtime > max(dic_mtime, aff_mtime):
            if not quiet:
                print(f"  Загрузка из кэша {cache_path.name}...")
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass  # Если кэш повреждён, пересоздаём
    
    # Парсим словарь
    dictionary = HunspellDictionary(dic_path, aff_path, quiet=quiet)
    
    # Сохраняем в кэш
    if use_cache:
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(dictionary, f)
            if not quiet:
                print(f"  Кэш сохранён: {cache_path.name}")
        except Exception as e:
            if not quiet:
                print(f"  Не удалось сохранить кэш: {e}")
    
    return dictionary

#!/usr/bin/env python3
"""
INTERSLAVICFREQ — Help
Запустите: python readme_and_tests.py
"""

import sys
import os
import re

# ─── Определяем поддержку цветов ───
def _supports_color():
    """Проверяет поддержку ANSI цветов."""
    if os.environ.get('NO_COLOR'):
        return False
    if not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
        return False
    if sys.platform == 'win32':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
        except Exception:
            return False
    return True

USE_COLOR = _supports_color()

# ─── Цвета ───
if USE_COLOR:
    class C:
        RST = '\033[0m'
        DIM = '\033[90m'
        BOLD = '\033[1m'
        STR = '\033[33m'
        NUM = '\033[35m'
        KW = '\033[31m'
        FN = '\033[36m'
        CMT = '\033[90m'
        ERR = '\033[91m'
        OK = '\033[32m'
else:
    class C:
        RST = DIM = BOLD = STR = NUM = KW = FN = CMT = ERR = OK = ''


def hl(code: str) -> str:
    """Подсветка синтаксиса (если поддерживается)."""
    if not USE_COLOR:
        return code
    lines = []
    for line in code.split('\n'):
        if '#' in line:
            idx = line.index('#')
            line = line[:idx] + C.CMT + line[idx:] + C.RST
        line = re.sub(r"('[^']*')", f'{C.STR}\\1{C.RST}', line)
        line = re.sub(r'("[^"]*")', f'{C.STR}\\1{C.RST}', line)
        for kw in ['import', 'from', 'as', 'True', 'False', 'None']:
            line = re.sub(rf'\b({kw})\b', f'{C.KW}\\1{C.RST}', line)
        line = re.sub(r'\b(\d+\.?\d*)\b', f'{C.NUM}\\1{C.RST}', line)
        line = re.sub(r'\b(\w+)\(', f'{C.FN}\\1{C.RST}(', line)
        lines.append(line)
    return '\n'.join(lines)


def format_result(result) -> str:
    """Форматирует результат для вывода."""
    if result is None:
        return ""
    if isinstance(result, float):
        return f"{result:.2f}"
    if isinstance(result, bool):
        color = C.OK if result else C.ERR
        return f"{color}{result}{C.RST}"
    if isinstance(result, list) and len(result) < 15:
        return repr(result)
    if isinstance(result, dict):
        return f"{{{len(result)} items}}"
    return repr(result)


def run(code: str, ctx: dict):
    """Показать и выполнить код с результатами на той же строке."""
    for line in code.strip().split('\n'):
        stripped = line.strip()
        
        # Пустые строки
        if not stripped:
            print()
            continue
        
        # Комментарии - просто печатаем
        if stripped.startswith('#'):
            print(hl(line))
            continue
        
        # Исполняемый код
        try:
            result = eval(stripped, ctx)
            res_str = format_result(result)
            if res_str:
                print(f"{hl(line)}  {C.CMT}# → {res_str}{C.RST}")
            else:
                print(hl(line))
        except SyntaxError:
            exec(stripped, ctx)
            print(hl(line))
        except Exception as e:
            print(f"{hl(line)}  {C.ERR}# Error: {e}{C.RST}")
    print()


# ─── MAIN ───
def show_help():
    """Show interactive documentation and demo for interslavicfreq library."""
    print(f"""
{C.BOLD}INTERSLAVICFREQ{C.RST}
A library for analyzing words and texts in Interslavic and other Slavic languages.
""")

    ctx = {}

    run("import interslavicfreq as isv", ctx)

    run("""
# Word frequency (Zipf scale: 3 = rare, 5+ = common)
isv.frequency('člověk')
isv.frequency('dom')
isv.frequency('xyz123')
""", ctx)

    run("""
# Full form: zipf_frequency(word, lang)
isv.zipf_frequency('dom', 'isv')
""", ctx)

    run("""
# Other languages
isv.frequency('człowiek', lang='pl')
isv.frequency('человек', lang='ru')
isv.frequency('člověk', lang='cs')
""", ctx)

    run("""
# Razumlivost — word intelligibility for Slavic speakers (0.0 - 1.0)
isv.razumlivost('dobro')
isv.razumlivost('prihoditi')
""", ctx)

    run("""
# Phrases: frequency = harmonic mean, razumlivost = arithmetic mean
isv.frequency('dobry denj')
isv.razumlivost('dobry denj')
""", ctx)

    run("""
# Spell checking
isv.spellcheck('prijatelj', 'isv')
isv.spellcheck('priyatel', 'isv')
""", ctx)

    run("""
# Percentage of correct words in a text
isv.correctness('Dobry denj, kako jesi?', 'isv')
isv.correctness('Dbory denj, kako jes?', 'isv')

""", ctx)

    run("""
# Tokenization
isv.simple_tokenize('Dobry denj!')
""", ctx)

    run("""
# Available dictionaries
isv.available_spellcheck_languages()
""", ctx)

    run("""
# Text quality index (weighted mean of frequency, razumlivost, correctness)
isv.quality_index('Dobry denj, kako jesi?')
isv.quality_index('Dobry denj, kako jesi?', frequency=0, razumlivost=0, correctness=1)
isv.quality_index('črnogledniki slusajut izvěstoglašenje')
""", ctx)

    run("""
# Phonetic similarity (~1.0 = identical, 0.0 = very different)
# Useful for cross-Slavic comparison and spelling variation
isv.phonetic_similarity('člověk', 'człowiek', lang='cs')
isv.phonetic_similarity('prijatelj', 'przyjaciel', lang='pl')
""", ctx)

    run("""
# Synonyms — find ISV synonyms for a word
isv.synonyms('mysliti')
isv.synonyms('dom')
""", ctx)

    run("""
# Best synonym — pick the best one by a scoring strategy
# best="frequency"    — highest Zipf frequency
# best="razumlivost"  — highest intelligibility score
# best="quality"      — highest quality_index (weighted combination)
isv.best_synonym('mysliti', best="frequency")
isv.best_synonym('mysliti', best="razumlivost")
isv.best_synonym('mysliti', best="quality")
""", ctx)

    run("""
# Reload synonyms without cache
isv.synonyms('mysliti', use_cache=False)
""", ctx)


if __name__ == '__main__':
    show_help()


# Alias for backward compatibility
main = show_help

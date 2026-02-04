Привет, у меня детективная история. 

Библиотека работала локально нормально, я сделал добавление на GitHub и установку через pip+link, и...

Код перенесся нормально. Файлы... Визуально тоже перенеслись полностью, есть все файлы со всеми названиями, содержание тоже полностью соответствует (правда, windows показывает разный объем, но я проверил, количество строчек одинаковое. да и не должен вроде git портить файлы. вроде).

Но у установленной библиотеки выскакивает ошибка # Error: 'HunspellDictionary' object has no attribute '_filter_isv_words'

И самое дебильное, проблема пропадает при замене папки hunspell на оригинальную. Хотя список файлов совпадает везде.




## interslavicfreq\__init__.py

```python
from __future__ import annotations

import gzip
import logging
import math
import random
import warnings
from functools import lru_cache
from typing import Iterator, Optional
from pathlib import Path

import msgpack

from .help import show_help as help

from .numbers import digit_freq, has_digit_sequence, smash_numbers
from .tokens import simple_tokenize, tokenize

from .util import data_path

logger = logging.getLogger(__name__)


CACHE_SIZE = 100000
DATA_PATH = data_path()

# Константы для нормализации Zipf в диапазон 0-1
ZIPF_MIN = 2.50
ZIPF_MAX = 7.33

# tokenize and simple_tokenize are imported so that other things can import
# them from here. Suppress the pyflakes warning.
tokenize = tokenize
simple_tokenize = simple_tokenize


def read_cBpack(filename: str) -> list[list[str]]:
    """
    Read a file from an idiosyncratic format that we use for storing
    approximate word frequencies, called "cBpack".

    The cBpack format is as follows:

    - The file on disk is a gzipped file in msgpack format, which decodes to a
      list whose first element is a header, and whose remaining elements are
      lists of words.

    - The header is a dictionary with 'format' and 'version' keys that make
      sure that we're reading the right thing.

    - Each inner list of words corresponds to a particular word frequency,
      rounded to the nearest centibel -- that is, one tenth of a decibel, or
      a factor of 10 ** .01.

      0 cB represents a word that occurs with probability 1, so it is the only
      word in the data (this of course doesn't happen). -200 cB represents a
      word that occurs once per 100 tokens, -300 cB represents a word that
      occurs once per 1000 tokens, and so on.

    - The index of each list within the overall list (without the header) is
      the negative of its frequency in centibels.

    - Each inner list is sorted in alphabetical order.

    As an example, consider a corpus consisting only of the words "red fish
    blue fish". The word "fish" occurs as 50% of tokens (-30 cB), while "red"
    and "blue" occur as 25% of tokens (-60 cB). The cBpack file of their word
    frequencies would decode to this:

        [
            {'format': 'cB', 'version': 1},
            [], [], [], ...    # 30 empty lists
            ['fish'],
            [], [], [], ...    # 29 more empty lists
            ['blue', 'red']
        ]
    """
    with gzip.open(filename, "rb") as infile:
        data = msgpack.load(infile, raw=False)
    header = data[0]
    if not isinstance(header, dict) or header.get("format") != "cB" or header.get("version") != 1:
        raise ValueError("Unexpected header: %r" % header)
    return data[1:]


def available_languages(wordlist: str = "best") -> dict[str, str]:
    """
    Given a wordlist name, return a dictionary of language codes to filenames,
    representing all the languages in which that wordlist is available.
    """
    if wordlist == "best":
        available = available_languages("small")
        available.update(available_languages("large"))
        return available
    elif wordlist == "combined":
        logger.warning("The 'combined' wordlists have been renamed to 'small'.")
        wordlist = "small"

    available = {}
    for path in DATA_PATH.glob("*.msgpack.gz"):
        if not path.name.startswith("_"):
            list_name = path.name.split(".")[0]
            name, lang = list_name.split("_")
            if name == wordlist:
                available[lang] = path
    return available


@lru_cache(maxsize=None)
def get_frequency_list(lang: str, wordlist: str = "best") -> list[list[str]]:
    """
    Read the raw data from a wordlist file, returning it as a list of
    lists. (See `read_cBpack` for what this represents.)
    """
    available = available_languages(wordlist)

    if lang not in available:
        raise LookupError(f"No wordlist {wordlist!r} available for language {lang!r}")

    return read_cBpack(available[lang])


def cB_to_freq(cB: int) -> float:
    """
    Convert a word frequency from the logarithmic centibel scale that we use
    internally, to a proportion from 0 to 1.

    On this scale, 0 cB represents the maximum possible frequency of
    1.0. -100 cB represents a word that happens 1 in 10 times,
    -200 cB represents something that happens 1 in 100 times, and so on.

    In general, x cB represents a frequency of 10 ** (x/100).
    """
    if cB > 0:
        raise ValueError("A frequency cannot be a positive number of centibels.")
    return 10 ** (cB / 100)


def cB_to_zipf(cB: int) -> float:
    """
    Convert a word frequency from centibels to the Zipf scale
    (see `zipf_to_freq`).

    The Zipf scale is related to centibels, the logarithmic unit that wordfreq
    uses internally, because the Zipf unit is simply the bel, with a different
    zero point. To convert centibels to Zipf, add 900 and divide by 100.
    """
    return (cB + 900) / 100


def zipf_to_freq(zipf: float) -> float:
    """
    Convert a word frequency from the Zipf scale to a proportion between 0 and
    1.

    The Zipf scale is a logarithmic frequency scale proposed by Marc Brysbaert,
    who compiled the SUBTLEX data. The goal of the Zipf scale is to map
    reasonable word frequencies to understandable, small positive numbers.

    A word rates as x on the Zipf scale when it occurs 10**x times per billion
    words. For example, a word that occurs once per million words is at 3.0 on
    the Zipf scale.
    """
    return 10**zipf / 1e9


def freq_to_zipf(freq: float) -> float:
    """
    Convert a word frequency from a proportion between 0 and 1 to the
    Zipf scale (see `zipf_to_freq`).
    """
    return math.log(freq, 10) + 9


@lru_cache(maxsize=None)
def get_frequency_dict(lang: str, wordlist: str = "best") -> dict[str, float]:
    """
    Get a word frequency list as a dictionary, mapping tokens to
    frequencies as floating-point probabilities.
    """
    freqs = {}
    pack = get_frequency_list(lang, wordlist)
    for index, bucket in enumerate(pack):
        freq = cB_to_freq(-index)
        for word in bucket:
            freqs[word] = freq
    return freqs


def iter_wordlist(lang: str, wordlist: str = "best") -> Iterator[str]:
    """
    Yield the words in a wordlist in approximate descending order of
    frequency.

    Because wordfreq rounds off its frequencies, the words will form 'bands'
    with the same rounded frequency, appearing in alphabetical order within
    each band.
    """
    for bucket in get_frequency_list(lang, wordlist):
        yield from bucket


# Cache for word_frequency
_wf_cache: dict[tuple[str, str, str, float], float] = {}


def _tokenize_for_freq(word: str, lang: str) -> list[str]:
    """
    Tokenize word for frequency lookup.
    Uses simple_tokenize for simplicity (no CJK handling needed for Slavic languages).
    """
    return simple_tokenize(word)


def _word_frequency(word: str, lang: str, wordlist: str, minimum: float) -> float:
    tokens = _tokenize_for_freq(word, lang)

    if not tokens:
        return minimum

    freqs = get_frequency_dict(lang, wordlist)
    
    # Для isvx (razumlivost) используем арифметическое среднее
    use_arithmetic_mean = lang == "isvx"
    
    if use_arithmetic_mean:
        total = 0.0
        count = 0
        for token in tokens:
            smashed = smash_numbers(token)
            if smashed not in freqs:
                return minimum
            token_freq = freqs[smashed]
            if smashed != token:
                token_freq *= digit_freq(token)
            total += token_freq
            count += 1
        freq = total / count if count > 0 else minimum
    else:
        # Гармоническое среднее (оригинальная логика)
        # 1 / f = 1 / f1 + 1 / f2 + ...
        one_over_result = 0.0
        for token in tokens:
            smashed = smash_numbers(token)
            if smashed not in freqs:
                return minimum
            token_freq = freqs[smashed]
            if smashed != token:
                token_freq *= digit_freq(token)
            one_over_result += 1.0 / token_freq
        freq = 1.0 / one_over_result

    # Округляем до 3 значащих цифр
    unrounded = max(freq, minimum)
    if unrounded == 0.0:
        return 0.0
    else:
        try:
            leading_zeroes = math.floor(-math.log(unrounded, 10))
        except ValueError:
            return minimum
        return round(unrounded, leading_zeroes + 3)


def word_frequency(word: str, lang: str, wordlist: str = "best", minimum: float = 0.0) -> float:
    """
    Get the frequency of `word` in the language with code `lang`, from the
    specified `wordlist`.

    These wordlists can be specified:

    - 'large': a wordlist built from at least 5 sources, containing word
      frequencies of 10^-8 and higher
    - 'small': a wordlist built from at least 3 sources, containing word
      frquencies of 10^-6 and higher
    - 'best': uses 'large' if available, and 'small' otherwise

    The value returned will always be at least as large as `minimum`.
    You could set this value to 10^-8, for example, to return 10^-8 for
    unknown words in the 'large' list instead of 0, avoiding a discontinuity.
    """
    args = (word, lang, wordlist, minimum)
    try:
        return _wf_cache[args]
    except KeyError:
        if len(_wf_cache) >= CACHE_SIZE:
            _wf_cache.clear()
        _wf_cache[args] = _word_frequency(*args)
        return _wf_cache[args]


def zipf_frequency(word: str, lang: str, wordlist: str = "best", minimum: float = 0.0) -> float:
    """
    Get the frequency of `word`, in the language with code `lang`, on the Zipf
    scale.

    The Zipf scale is a logarithmic frequency scale proposed by Marc Brysbaert,
    who compiled the SUBTLEX data. The goal of the Zipf scale is to map
    reasonable word frequencies to understandable, small positive numbers.

    A word rates as x on the Zipf scale when it occurs 10**x times per billion
    words. For example, a word that occurs once per million words is at 3.0 on
    the Zipf scale.

    Zipf values for reasonable words are between 0 and 8. The value this
    function returns will always be at last as large as `minimum`, even for a
    word that never appears. The default minimum is 0, representing words
    that appear once per billion words or less.

    wordfreq internally quantizes its frequencies to centibels, which are
    1/100 of a Zipf unit. The output of `zipf_frequency` will be rounded to
    the nearest hundredth to match this quantization.
    """
    freq_min = zipf_to_freq(minimum)
    freq = word_frequency(word, lang, wordlist, freq_min)
    return round(freq_to_zipf(freq), 2)


def frequency(word: str, lang: str = "isv", wordlist: str = "best", minimum: float = 0.0) -> float:
    """
    Alias for zipf_frequency with default lang='isv'.
    
    Get the frequency of `word` on the Zipf scale.
    """
    return zipf_frequency(word, lang, wordlist, minimum)


def razumlivost(word: str, wordlist: str = "best", minimum: float = 0.0) -> float:
    """
    Get intelligibility score (razumlivost) for a word.
    Uses arithmetic mean for multi-token phrases.
    """
    return zipf_frequency(word, "isvx", wordlist, minimum)


def _normalize_zipf(zipf: float) -> float:
    """
    Normalize Zipf value to 0-1 range.
    
    Uses ZIPF_MIN (2.50) and ZIPF_MAX (7.33) as bounds.
    Values outside bounds are clamped to 0.0 or 1.0.
    """
    if zipf <= ZIPF_MIN:
        return 0.0
    if zipf >= ZIPF_MAX:
        return 1.0
    return (zipf - ZIPF_MIN) / (ZIPF_MAX - ZIPF_MIN)


def quality_index(
    text: str,
    frequency: float = 0.33,
    razumlivost: float = 0.33,
    correctness: float = 0.33
) -> float:
    """
    Calculate overall quality index for Interslavic text.
    
    Uses weighted average of three metrics:
    - frequency: normalized Zipf frequency (0-1)
    - razumlivost: intelligibility score (0-1)  
    - correctness: spelling correctness (0-1)
    
    Args:
        text: The text to analyze
        frequency: Weight for frequency metric (default: 0.33)
        razumlivost: Weight for razumlivost metric (default: 0.33)
        correctness: Weight for correctness metric (default: 0.33)
    
    Returns:
        A value between 0.0 and 1.0 representing overall quality.
    """
    # Получаем значения метрик
    freq_zipf = zipf_frequency(text, 'isv')
    razum_val = globals()['razumlivost'](text)  # Avoid name collision with parameter
    corr_val = globals()['correctness'](text, 'isv')
    
    # Нормализуем frequency из Zipf в 0-1
    freq_norm = _normalize_zipf(freq_zipf)
    
    # Взвешенное среднее
    total_weight = frequency + razumlivost + correctness
    if total_weight == 0:
        return 0.0
    
    return (freq_norm * frequency + razum_val * razumlivost + corr_val * correctness) / total_weight


@lru_cache(maxsize=100)
def top_n_list(lang: str, n: int, wordlist: str = "best", ascii_only: bool = False) -> list[str]:
    """
    Return a frequency list of length `n` in descending order of frequency.
    This list contains words from `wordlist`, of the given language.
    If `ascii_only`, then only ascii words are considered.

    The frequency list will not contain multi-digit sequences, because we
    estimate the frequencies of those using the functions in `numbers.py`,
    not using a wordlist that contains all of them.
    """
    results = []
    for word in iter_wordlist(lang, wordlist):
        if (not ascii_only) or max(word) <= "~":
            if not has_digit_sequence(word):
                results.append(word)
                if len(results) >= n:
                    break
    return results


def random_words(
    lang: str = "en",
    wordlist: str = "best",
    nwords: int = 5,
    bits_per_word: int = 12,
    ascii_only: bool = False,
) -> str:
    """
    Returns a string of random, space separated words.

    These words are of the given language and from the given wordlist.
    There will be `nwords` words in the string.

    `bits_per_word` determines the amount of entropy provided by each word;
    when it's higher, this function will choose from a larger list of
    words, some of which are more rare.

    You can restrict the selection of words to those written in ASCII
    characters by setting `ascii_only` to True.
    """
    n_choices = 2**bits_per_word
    choices = top_n_list(lang, n_choices, wordlist, ascii_only=ascii_only)
    if len(choices) < n_choices:
        raise ValueError(
            "There aren't enough words in the wordlist to provide %d bits of "
            "entropy per word." % bits_per_word
        )
    return " ".join([random.choice(choices) for i in range(nwords)])


def random_ascii_words(
    lang: str = "en", wordlist: str = "best", nwords: int = 5, bits_per_word: int = 12
) -> str:
    """
    Returns a string of random, space separated, ASCII words.

    These words are of the given language and from the given wordlist.
    There will be `nwords` words in the string.

    `bits_per_word` determines the amount of entropy provided by each word;
    when it's higher, this function will choose from a larger list of
    words, some of which are more rare.
    """
    return random_words(lang, wordlist, nwords, bits_per_word, ascii_only=True)


# ============================================================================
# Hunspell spellcheck integration
# ============================================================================

from .spellcheck_hunspell import load_hunspell, HunspellDictionary

_hunspell_cache: dict[str, HunspellDictionary] = {}


def _get_hunspell_dir() -> Path:
    """Get the path to the Hunspell dictionaries directory."""
    return data_path().parent / "hunspell"


def available_spellcheck_languages() -> list[str]:
    """
    Get list of available languages for spellchecking.
    Languages are determined by .dic files in the hunspell directory.
    """
    hunspell_dir = _get_hunspell_dir()
    if not hunspell_dir.exists():
        return []
    return sorted([
        p.stem for p in hunspell_dir.glob("*.dic")
        if (hunspell_dir / f"{p.stem}.aff").exists()
    ])


def _get_hunspell_dict(lang: str) -> HunspellDictionary:
    """Get or load a Hunspell dictionary for the given language."""
    if lang not in _hunspell_cache:
        hunspell_dir = _get_hunspell_dir()
        dic_path = hunspell_dir / f"{lang}.dic"
        aff_path = hunspell_dir / f"{lang}.aff"
        
        if not dic_path.exists() or not aff_path.exists():
            available = available_spellcheck_languages()
            raise LookupError(
                f"No Hunspell dictionary for language {lang!r}. "
                f"Available: {available}"
            )
        
        _hunspell_cache[lang] = load_hunspell(dic_path, aff_path, quiet=True)
    
    return _hunspell_cache[lang]


def spellcheck(word: str, lang: str) -> bool:
    """
    Check if a word is spelled correctly.
    
    Args:
        word: The word to check
        lang: Language code (e.g., 'isv', 'ru', 'pl')
    
    Returns:
        True if the word is spelled correctly, False otherwise.
    """
    dictionary = _get_hunspell_dict(lang)
    return dictionary.check(word)


def correctness(text: str, lang: str = "isv") -> float:
    """
    Calculate the percentage of correctly spelled words in the text.
    Uses arithmetic mean (correct_words / total_words).
    
    Args:
        text: The text to check
        lang: Language code (e.g., 'isv', 'ru', 'pl')
    
    Returns:
        A value between 0.0 and 1.0 representing the proportion of
        correctly spelled words.
    """
    tokens = simple_tokenize(text)
    
    if not tokens:
        return 1.0
    
    dictionary = _get_hunspell_dict(lang)
    correct_count = sum(1 for token in tokens if dictionary.check(token))
    
    return correct_count / len(tokens)

# Show import notice
print("Call interslavicfreq.help() or isv.help() to learn about library features")

```

## interslavicfreq\help.py

```python
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
Библиотека анализа слов и текстов для межславянского и других славянских языков.
""")

    ctx = {}

    run("import interslavicfreq as isv", ctx)

    run("""
# Частота слова (шкала Zipf: 3 = редко, 5+ = часто)
isv.frequency('člověk')
isv.frequency('dom')
isv.frequency('xyz123')
""", ctx)

    run("""
# Полная форма: zipf_frequency(word, lang)
isv.zipf_frequency('dom', 'isv')
""", ctx)

    run("""
# Другие языки
isv.frequency('człowiek', lang='pl')
isv.frequency('человек', lang='ru')
isv.frequency('člověk', lang='cs')
""", ctx)

    run("""
# Razumlivost — понятность слова для славян (0.0 - 1.0)
isv.razumlivost('dobro')
isv.razumlivost('prihoditi')
""", ctx)

    run("""
# Фразы: frequency = гармоническое среднее, razumlivost = арифметическое
isv.frequency('dobry denj')
isv.razumlivost('dobry denj')
""", ctx)

    run("""
# Проверка орфографии
isv.spellcheck('prijatelj', 'isv')
isv.spellcheck('priyatel', 'isv')
""", ctx)

    run("""
# Процент корректных слов в тексте
isv.correctness('Dobry denj, kako jesi?', 'isv')
isv.correctness('Dbory denj, kako jes?', 'isv')

""", ctx)

    run("""
# Токенизация
isv.simple_tokenize('Dobry denj!')
""", ctx)

    run("""
# Доступные словари
isv.available_spellcheck_languages()
""", ctx)

    run("""
# Индекс качества текста (взвешенное среднее frequency, razumlivost, correctness)
isv.quality_index('Dobry denj, kako jesi?')
isv.quality_index('Dobry denj, kako jesi?', frequency=0, razumlivost=0, correctness=1)
isv.quality_index('črnogledniki slusajut izvěstoglašenje')
""", ctx)

if __name__ == '__main__':
    show_help()


# Alias for backward compatibility
main = show_help

```

## interslavicfreq\language_info.py

```python
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
```

## interslavicfreq\numbers.py

```python
import regex

# Frequencies of leading digits, according to Benford's law, sort of.
# Benford's law doesn't describe numbers with leading zeroes, because "007"
# and "7" are the same number, but for us they should have different frequencies.
# I added an estimate for the frequency of numbers with leading zeroes.
DIGIT_FREQS = [0.009, 0.300, 0.175, 0.124, 0.096, 0.078, 0.066, 0.057, 0.050, 0.045]

# Suppose you have a token NNNN, a 4-digit number representing a year. We're making
# a probability distribution of P(token=NNNN) | P(token is 4 digits).
#
# We do this with a piecewise exponential function whose peak is a plateau covering
# the years 2019 to 2039.

# Determined by experimentation: makes the probabilities of all years add up to 90%.
# The other 10% goes to NOT_YEAR_PROB. tests/test_numbers.py confirms that this
# probability distribution adds up to 1.
YEAR_LOG_PEAK = -1.9185
NOT_YEAR_PROB = 0.1
REFERENCE_YEAR = 2019
PLATEAU_WIDTH = 20

DIGIT_RE = regex.compile(r"\d")
MULTI_DIGIT_RE = regex.compile(r"\d[\d.,]+")
PURE_DIGIT_RE = regex.compile(r"\d+")


def benford_freq(text: str) -> float:
    """
    Estimate the frequency of a digit sequence according to Benford's law.
    """
    first_digit = int(text[0])
    return DIGIT_FREQS[first_digit] / 10 ** (len(text) - 1)


def year_freq(text: str) -> float:
    """
    Estimate the relative frequency of a particular 4-digit sequence representing
    a year.

    For example, suppose text == "1985". We're estimating the probability that a
    randomly-selected token from a large corpus will be "1985" and refer to the
    year, _given_ that it is 4 digits. Tokens that are not 4 digits are not involved
    in the probability distribution.
    """
    year = int(text)

    # Fitting a line to the curve seen at
    # https://twitter.com/r_speer/status/1493715982887571456.

    if year <= REFERENCE_YEAR:
        year_log_freq = YEAR_LOG_PEAK - 0.0083 * (REFERENCE_YEAR - year)

    # It's no longer 2019, which is when the Google Books data was last collected.
    # It's 2022 as I write this, and possibly even later as you're using it. Years
    # keep happening.
    #
    # So, we'll just keep the expected frequency of the "present" year constant for
    # 20 years.

    elif REFERENCE_YEAR < year <= REFERENCE_YEAR + PLATEAU_WIDTH:
        year_log_freq = YEAR_LOG_PEAK

    # Fall off quickly to catch up with the actual frequency of future years
    # (it's low). This curve is made up to fit with the made-up "present" data above.
    else:
        year_log_freq = YEAR_LOG_PEAK - 0.2 * (year - (REFERENCE_YEAR + PLATEAU_WIDTH))

    year_prob = 10.0**year_log_freq

    # If this token _doesn't_ represent a year, then use the Benford frequency
    # distribution.
    not_year_prob = NOT_YEAR_PROB * benford_freq(text)
    return year_prob + not_year_prob


def digit_freq(text: str) -> float:
    """
    Get the relative frequency of a string of digits, using our estimates.
    """
    freq = 1.0
    for match in MULTI_DIGIT_RE.findall(text):
        for submatch in PURE_DIGIT_RE.findall(match):
            if len(submatch) == 4:
                freq *= year_freq(submatch)
            else:
                freq *= benford_freq(submatch)
    return freq


def has_digit_sequence(text: str) -> bool:
    """
    Returns True iff the text has a digit sequence that will be normalized out
    and handled with `digit_freq`.
    """
    return bool(MULTI_DIGIT_RE.match(text))


def _sub_zeroes(match: regex.Match) -> str:
    """
    Given a regex match, return what it matched with digits replaced by
    zeroes.
    """
    return DIGIT_RE.sub("0", match.group(0))


def smash_numbers(text: str) -> str:
    """
    Replace sequences of multiple digits with zeroes, so we don't need to
    distinguish the frequencies of thousands of numbers.
    """
    return MULTI_DIGIT_RE.sub(_sub_zeroes, text)

```

## interslavicfreq\preprocess.py

```python
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

```

## interslavicfreq\spellcheck_hunspell.py

```python
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

def _get_cache_dir() -> Path:
    """
    Get the cache directory for Hunspell dictionaries.
    Uses standard user cache locations for each OS.
    """
    app_name = "interslavicfreq"
    
    if sys.platform == "win32":
        # Windows: %LOCALAPPDATA% or %APPDATA%
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            cache_dir = Path(base) / app_name / "cache"
        else:
            cache_dir = Path.home() / ".cache" / app_name
    elif sys.platform == "darwin":
        # macOS: ~/Library/Caches/
        cache_dir = Path.home() / "Library" / "Caches" / app_name
    else:
        # Linux/Unix: $XDG_CACHE_HOME or ~/.cache/
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache:
            cache_dir = Path(xdg_cache) / app_name
        else:
            cache_dir = Path.home() / ".cache" / app_name
    
    # Создаем папку, если её нет
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_cache_path(dic_path: Path) -> Path:
    """Получить путь к файлу кэша в пользовательской папке."""
    cache_dir = _get_cache_dir()
    # Создаем имя файла на основе исходного словаря, чтобы не было конфликтов
    cache_name = f"{dic_path.stem}.cache.pickle"
    return cache_dir / cache_name

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

```

## interslavicfreq\tokens.py

```python
from __future__ import annotations

import unicodedata

import regex

# Для совместимости со старым кодом
SPACELESS_SCRIPTS: list[str] = []
EXTRA_JAPANESE_CHARACTERS = ""


def _make_spaceless_expr() -> str:
    # Упрощено: для славянских языков не нужны spaceless scripts
    return r"\p{IsIdeo}"


SPACELESS_EXPR = _make_spaceless_expr()

# All vowels that might appear at the start of a word in French or Catalan,
# plus 'h' which would be silent and imply a following vowel sound.
INITIAL_VOWEL_EXPR = "[AEHIOUYÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÅÏÖŒaehiouyáéíóúàèìòùâêîôûåïöœ]"

TOKEN_RE = regex.compile(
    r"""
    # Case 1: a special case for non-spaced languages
    # -----------------------------------------------

    # Some scripts are written without spaces, and the Unicode algorithm
    # seems to overreact and insert word breaks between all their letters.
    # When we see sequences of characters in these scripts, we make sure not
    # to break them up. Such scripts include Han ideographs (\p{IsIdeo}),
    # hiragana (\p{Script=Hiragana}), and many Southeast Asian scripts such
    # as Thai and Khmer.
    #
    # Without this case, the standard rule (case 2) would make each character
    # a separate token. This would be the correct behavior for word-wrapping,
    # but a messy failure mode for NLP tokenization.
    #
    # If you have Chinese or Japanese text, it's certainly better to use a
    # tokenizer that's designed for it. Elsewhere in this file, we have
    # specific tokenizers that can handle Chinese and Japanese. With this
    # rule, though, at least this general tokenizer will fail less badly
    # on those languages.
    #
    # This rule is listed first so that it takes precedence. The placeholder
    # <SPACELESS> will be replaced by the complex range expression made by
    # _make_spaceless_expr().

    [<SPACELESS>]+
    |

    # Case 2: Gender-neutral "@s"
    # ---------------------------
    #
    # "@" and "@s" are gender-neutral word endings that can replace -a, -o,
    # -as, and -os in Spanish, Portuguese, and occasionally Italian.
    #
    # This doesn't really conflict with other uses of the @ sign, so we simply
    # recognize these endings as being part of the token in any language.
    #
    # We will recognize the endings as part of our main rule for recognizing
    # words, which is Case 3 below. However, one case that remains separate is
    # the Portuguese word "@s" itself, standing for the article "as" or "os".
    # This must be followed by a word break (\b).

    @s \b
    |

    # Case 3: Unicode segmentation with tweaks
    # ----------------------------------------

    # The start of the token must be 'word-like', not punctuation or whitespace
    # or various other things. However, we allow characters of category So
    # (Symbol - Other) because many of these are emoji, which can convey
    # meaning.

    (?=[\w\p{So}])

    # The start of the token must not consist of 1-2 letters, an apostrophe,
    # and a vowel or 'h'. This is a sequence that occurs particularly in French
    # phrases such as "l'arc", "d'heure", or "qu'un". In these cases we want
    # the sequence up to the apostrophe to be considered as a separate token,
    # even though apostrophes are not usually word separators (the word "won't"
    # does not separate into "won" and "t").
    #
    # This would be taken care of by optional rule "WB5a" in Unicode TR29,
    # "Unicode Text Segmentation". That optional rule was applied in `regex`
    # before June 2018, but no longer is, so we have to do it ourselves.

    (?!\w\w?'<VOWEL>)

    # The entire token is made of graphemes (\X). Matching by graphemes means
    # that we don't have to specially account for marks or ZWJ sequences. We
    # use a non-greedy match so that we can control where the match ends in the
    # following expression.
    #
    # If we were matching by codepoints (.) instead of graphemes (\X), then
    # detecting boundaries would be more difficult. Here's a fact about the
    # regex module that's subtle and poorly documented: a position that's
    # between codepoints, but in the middle of a grapheme, does not match as a
    # word break (\b), but also does not match as not-a-word-break (\B). The
    # word boundary algorithm simply doesn't apply in such a position. It is
    # unclear whether this is intentional.

    \X+?

    # The token ends when it encounters a word break (\b). We use the
    # non-greedy match (+?) to make sure to end at the first word break we
    # encounter.
    #
    # We need a special case for gender-neutral "@", which is acting as a
    # letter, but Unicode considers it to be a symbol and would break words
    # around it.  We prefer continuing the token with "@" or "@s" over matching
    # a word break.
    #
    # As in case 2, this is only allowed at the end of the word. Unfortunately,
    # we can't use the word-break expression \b in this case, because "@"
    # already is a word break according to Unicode. Instead, we use a negative
    # lookahead assertion to ensure that the next character is not word-like.
    (?:
       @s? (?!\w) | \b
    )
    |

    # Another subtle fact: the "non-breaking space" U+A0 counts as a word break
    # here. That's surprising, but it's also what we want, because we don't want
    # any kind of spaces in the middle of our tokens.

    # Case 4: Match French apostrophes
    # --------------------------------
    # This allows us to match the particles in French, Catalan, and related
    # languages, such as «l'» and «qu'», that we may have excluded from being
    # part of the token in Case 3.

    \w\w?'
""".replace("<SPACELESS>", SPACELESS_EXPR).replace("<VOWEL>", INITIAL_VOWEL_EXPR),
    regex.V1 | regex.WORD | regex.VERBOSE,
)

TOKEN_RE_WITH_PUNCTUATION = regex.compile(
    r"""
    # This expression is similar to the expression above. It adds a case between
    # 2 and 3 that matches any sequence of punctuation characters.

    [<SPACELESS>]+ |                                        # Case 1
    @s \b |                                                 # Case 2
    [\p{punct}]+ |                                          # punctuation
    (?=[\w\p{So}]) (?!\w\w?'<VOWEL>)
      \X+? (?: @s? (?!w) | \b) |                            # Case 3
    \w\w?'                                                  # Case 4
""".replace("<SPACELESS>", SPACELESS_EXPR).replace("<VOWEL>", INITIAL_VOWEL_EXPR),
    regex.V1 | regex.WORD | regex.VERBOSE,
)



def simple_tokenize(text: str, include_punctuation: bool = False) -> list[str]:
    """
    Tokenize the given text using a straightforward, Unicode-aware token
    expression.

    The expression mostly implements the rules of Unicode Annex #29 that
    are contained in the `regex` module's word boundary matching, including
    the refinement that splits words between apostrophes and vowels in order
    to separate tokens such as the French article «l'».

    It makes sure not to split in the middle of a grapheme, so that zero-width
    joiners and marks on Devanagari words work correctly.

    Our customizations to the expression are:

    - It leaves sequences of Chinese or Japanese characters (specifically, Han
      ideograms and hiragana) relatively untokenized, instead of splitting each
      character into its own token.

    - If `include_punctuation` is False (the default), it outputs only the
      tokens that start with a word-like character, or miscellaneous symbols
      such as emoji. If `include_punctuation` is True, it outputs all non-space
      tokens.

    - It keeps Southeast Asian scripts, such as Thai, glued together. This yields
      tokens that are much too long, but the alternative is that every grapheme
      would end up in its own token, which is worse.
    """
    text = unicodedata.normalize("NFC", text)
    if include_punctuation:
        return [token.casefold() for token in TOKEN_RE_WITH_PUNCTUATION.findall(text)]
    else:
        return [token.strip("'").casefold() for token in TOKEN_RE.findall(text)]


def tokenize(
    text: str,
    lang: str = "isv",
    include_punctuation: bool = False,
) -> list[str]:
    """
    Tokenize text for Slavic languages.
    
    Simplified version without CJK support.
    Uses Unicode-aware regex tokenization with NFC normalization.

    Args:
        text: Text to tokenize
        lang: Language code (default: "isv")
        include_punctuation: Whether to include punctuation tokens
    
    Returns:
        List of tokens
    """
    # NFC normalization (подходит для латиницы и кириллицы)
    text = unicodedata.normalize("NFC", text)
    
    # Casefold
    text = text.casefold()
    
    return simple_tokenize(text, include_punctuation=include_punctuation)


def lossy_tokenize(
    text: str,
    lang: str,
    include_punctuation: bool = False,
) -> list[str]:
    """
    Tokenize with lossy normalization for frequency lookup.
    
    Simplified version for Slavic languages.
    """
    try:
        from ftfy.fixes import uncurl_quotes
        tokens = tokenize(text, lang, include_punctuation)
        return [uncurl_quotes(token) for token in tokens]
    except ImportError:
        # ftfy is optional
        return tokenize(text, lang, include_punctuation)
```

## interslavicfreq\transliterate.py

```python
from __future__ import annotations

# This table comes from
# https://github.com/opendatakosovo/cyrillic-transliteration/blob/master/cyrtranslit/mapping.py,
# from the 'cyrtranslit' module. We originally had to reimplement it because
# 'cyrtranslit' didn't work in Python 3; now it does, but we've made the table
# more robust than the one in cyrtranslit.
SR_LATN_TABLE = {
    ord("А"): "A",
    ord("а"): "a",
    ord("Б"): "B",
    ord("б"): "b",
    ord("В"): "V",
    ord("в"): "v",
    ord("Г"): "G",
    ord("г"): "g",
    ord("Д"): "D",
    ord("д"): "d",
    ord("Ђ"): "Đ",
    ord("ђ"): "đ",
    ord("Е"): "E",
    ord("е"): "e",
    ord("Ж"): "Ž",
    ord("ж"): "ž",
    ord("З"): "Z",
    ord("з"): "z",
    ord("И"): "I",
    ord("и"): "i",
    ord("Ј"): "J",
    ord("ј"): "j",
    ord("К"): "K",
    ord("к"): "k",
    ord("Л"): "L",
    ord("л"): "l",
    ord("Љ"): "Lj",
    ord("љ"): "lj",
    ord("М"): "M",
    ord("м"): "m",
    ord("Н"): "N",
    ord("н"): "n",
    ord("Њ"): "Nj",
    ord("њ"): "nj",
    ord("О"): "O",
    ord("о"): "o",
    ord("П"): "P",
    ord("п"): "p",
    ord("Р"): "R",
    ord("р"): "r",
    ord("С"): "S",
    ord("с"): "s",
    ord("Т"): "T",
    ord("т"): "t",
    ord("Ћ"): "Ć",
    ord("ћ"): "ć",
    ord("У"): "U",
    ord("у"): "u",
    ord("Ф"): "F",
    ord("ф"): "f",
    ord("Х"): "H",
    ord("х"): "h",
    ord("Ц"): "C",
    ord("ц"): "c",
    ord("Ч"): "Č",
    ord("ч"): "č",
    ord("Џ"): "Dž",
    ord("џ"): "dž",
    ord("Ш"): "Š",
    ord("ш"): "š",
    # Handle Cyrillic letters from other languages. We hope these cases don't
    # come up often when we're trying to transliterate Serbian, but if these
    # letters show up in loan-words or code-switching text, we can at least
    # transliterate them approximately instead of leaving them as Cyrillic
    # letters surrounded by Latin.
    # Russian letters
    ord("Ё"): "Jo",
    ord("ё"): "jo",
    ord("Й"): "J",
    ord("й"): "j",
    ord("Щ"): "Šč",
    ord("щ"): "šč",
    ord("Ъ"): "",
    ord("ъ"): "",
    ord("Ы"): "Y",
    ord("ы"): "y",
    ord("Ь"): "'",
    ord("ь"): "'",
    ord("Э"): "E",
    ord("э"): "e",
    ord("Ю"): "Ju",
    ord("ю"): "ju",
    ord("Я"): "Ja",
    ord("я"): "ja",
    # Belarusian letter
    ord("Ў"): "Ŭ",
    ord("ў"): "ŭ",
    # Ukrainian letters
    ord("Є"): "Je",
    ord("є"): "je",
    ord("І"): "I",
    ord("і"): "i",
    ord("Ї"): "Ï",
    ord("ї"): "ï",
    ord("Ґ"): "G",
    ord("ґ"): "g",
    # Macedonian letters
    ord("Ѕ"): "Dz",
    ord("ѕ"): "dz",
    ord("Ѓ"): "Ǵ",
    ord("ѓ"): "ǵ",
    ord("Ќ"): "Ḱ",
    ord("ќ"): "ḱ",
}

AZ_LATN_TABLE = SR_LATN_TABLE.copy()
AZ_LATN_TABLE.update(
    {
        # Distinct Azerbaijani letters
        ord("Ҹ"): "C",
        ord("ҹ"): "c",
        ord("Ә"): "Ə",
        ord("ә"): "ə",
        ord("Ғ"): "Ğ",
        ord("ғ"): "ğ",
        ord("Һ"): "H",
        ord("һ"): "h",
        ord("Ө"): "Ö",
        ord("ө"): "ö",
        ord("Ҝ"): "G",
        ord("ҝ"): "g",
        ord("Ү"): "Ü",
        ord("ү"): "ü",
        # Azerbaijani letters with different transliterations
        ord("Ч"): "Ç",
        ord("ч"): "ç",
        ord("Х"): "X",
        ord("х"): "x",
        ord("Ы"): "I",
        ord("ы"): "ı",
        ord("И"): "İ",
        ord("и"): "i",
        ord("Ж"): "J",
        ord("ж"): "j",
        ord("Ј"): "Y",
        ord("ј"): "y",
        ord("Г"): "Q",
        ord("г"): "q",
        ord("Ш"): "Ş",
        ord("ш"): "ş",
    }
)


def transliterate(table: dict[int, str], text: str) -> str:
    """
    Transliterate text according to one of the tables above.

    `table` chooses the table. It looks like a language code but comes from a
    very restricted set:

    - 'sr-Latn' means to convert Serbian, which may be in Cyrillic, into the
      Latin alphabet.
    - 'az-Latn' means the same for Azerbaijani Cyrillic to Latn.
    """
    if table == "sr-Latn":
        return text.translate(SR_LATN_TABLE)
    elif table == "az-Latn":
        return text.translate(AZ_LATN_TABLE)
    else:
        raise ValueError(f"Unknown transliteration table: {table!r}")

```

## interslavicfreq\util.py

```python
from __future__ import annotations
from pathlib import Path

def data_path(filename: str | None = None) -> Path:
    """
    Get a path to a file in the data directory.
    """
    base = Path(__file__).parent
    if filename is None:
        return base / "data" / "frequency"
    return base / "data" / "frequency" / filename
```

## LICENSE.md

```markdown
MIT License

Copyright (c) 2026 gorlatoff

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```

## pyproject.toml

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "interslavicfreq"
version = "0.1.0"
authors = [
  { name="Mikhail Gorlatov", email="gorlatoff@gmail.com" },
]
description = "A library for word and text analysis for Interslavic and other Slavic languages. Forked from wordfreq."
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
keywords = ["interslavic", "slavic", "linguistics", "wordfreq", "nlp", "zipf"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Topic :: Text Processing :: Linguistic",
]
dependencies = [
    "msgpack",
    "regex",
]

[project.urls]
"Homepage" = "https://github.com/gorlatoff/interslavicfreq"
"Source" = "https://github.com/gorlatoff/interslavicfreq"

[tool.setuptools]
include-package-data = true

[tool.setuptools.package-data]
# Это ГАРАНТИРУЕТ включение файлов данных (частотности и словари)
"interslavicfreq" = ["data/**/*"]

[tool.setuptools.packages.find]
where = ["."]
include = ["interslavicfreq*"]
```

## README.md

```markdown
📌 INTERSLAVICFREQ

interslavicfreq is a Python library for word and text analysis of the Interslavic language (Medžuslovjanski) and other Slavic languages. It allows for frequency estimation, intelligibility (razumlivost) scoring, and text quality assessment.

> Note: This project is a fork of the wordfreq (https://github.com/rspeer/wordfreq) library, specifically modified for Slavic linguistics.

✏️ Installation

pip install git+https://github.com/medzuslovjansky/interslavicfreq.git

✏️ Usage Examples

```python
import interslavicfreq as isv

# Word frequency (Zipf scale: 3 = rare, 5+ = frequent)
isv.frequency('člověk')  # → 5.84
isv.frequency('dom')  # → 5.22
isv.frequency('xyz123')  # → 0.00

# Full form: zipf_frequency(word, lang)
isv.zipf_frequency('dom', 'isv')  # → 5.22

# Other languages
isv.frequency('człowiek', lang='pl')  # → 5.36
isv.frequency('человек', lang='ru')  # → 5.96
isv.frequency('člověk', lang='cs')  # → 5.57

# Razumlivost — word intelligibility for Slavs (0.0 - 1.0)
isv.razumlivost('dobro')  # → 0.85
isv.razumlivost('prihoditi')  # → 0.77

# Phrases: frequency = harmonic mean, razumlivost = arithmetic mean
isv.frequency('dobry denj')  # → 5.54
isv.razumlivost('dobry denj')  # → 0.83

# Spellcheck
isv.spellcheck('prijatelj', 'isv')  # → True
isv.spellcheck('priyatel', 'isv')  # → False

# Percentage of correct words in the text
isv.correctness('Dobry denj, kako jesi?', 'isv')  # → 1.00
isv.correctness('Dbory denj, kako jesteś?', 'isv')  # → 0.50

# Tokenization
isv.simple_tokenize('Dobry denj!')  # → ['dobry', 'denj']

# Available dictionaries
isv.available_spellcheck_languages()  # → ['be', 'bg', 'cs', 'en', 'hr', 'isv', 'mk', 'pl', 'ru', 'sk', 'sl', 'sr', 'uk']

# Text quality index (weighted average of frequency, razumlivost, correctness)
isv.quality_index('Dobry denj, kako jesi?')  # → 0.81
isv.quality_index('Dobry denj, kako jesi?', frequency=0, razumlivost=0, correctness=1)  # → 1.00
isv.quality_index('črnogledniki slusajut izvěstoglašenje')  # → 0.22

```

✏️ Requirements
• Tested on Python 3.14.

✏️ License
This project is licensed under the MIT License.

✏️ Author
Mikhail Gorlatov - gorlatoff@gmail.com
```


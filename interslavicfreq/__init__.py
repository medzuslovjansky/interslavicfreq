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

from .isv_phonetic_distance import phonetic_distance, phonetic_similarity

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
print("Try `interslavicfreq.help()` to learn about library features (or `isv.help()` if `import as isv`)")

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

# Synonyms — find ISV synonyms for a word
isv.synonyms('mysliti')  # → {'mysliti', 'mněvati', 'mněti'}
isv.synonyms('dom')  # → {'dom'}

# Best synonym — pick the best one by a scoring strategy
# best="frequency"    — highest Zipf frequency
# best="razumlivost"  — highest intelligibility score
# best="quality"      — highest quality_index (weighted combination)
isv.best_synonym('mysliti', best="frequency")  # → 'mysliti'
isv.best_synonym('mysliti', best="razumlivost")  # → 'mysliti'
isv.best_synonym('mysliti', best="quality")  # → 'mysliti'

# Reload synonyms without cache
isv.synonyms('mysliti', use_cache=False)  # → {'mysliti', 'mněvati', 'mněti'}
```

✏️ Requirements
• Tested on Python 3.14.

✏️ License
This project is licensed under the MIT License.

✏️ Author
Mikhail Gorlatov - gorlatoff@gmail.com
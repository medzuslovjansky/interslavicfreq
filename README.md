📌 INTERSLAVICFREQ

interslavicfreq is a Python library for word and text analysis of the Interslavic language (Medžuslovjanski) and other Slavic languages. It allows for frequency estimation, intelligibility (razumlivost) scoring, and text quality assessment.

> Note: This project is a fork of the wordfreq (https://github.com/rspeer/wordfreq) library, specifically modified for Slavic linguistics.

✏️ Installation

pip install git+https://github.com/gorlatoff/interslavicfreq.git

✏️ Usage Examples

```python
import interslavicfreq as isv

# Частота слова (шкала Zipf: 3 = редко, 5+ = часто)
isv.frequency('člověk')  # → 5.84
isv.frequency('dom')  # → 5.22
isv.frequency('xyz123')  # → 0.00

# Полная форма: zipf_frequency(word, lang)
isv.zipf_frequency('dom', 'isv')  # → 5.22

# Другие языки
isv.frequency('człowiek', lang='pl')  # → 5.36
isv.frequency('человек', lang='ru')  # → 5.96
isv.frequency('člověk', lang='cs')  # → 5.57

# Razumlivost — понятность слова для славян (0.0 - 1.0)
isv.razumlivost('dobro')  # → 0.85
isv.razumlivost('prihoditi')  # → 0.77

# Фразы: frequency = гармоническое среднее, razumlivost = арифметическое
isv.frequency('dobry denj')  # → 5.54
isv.razumlivost('dobry denj')  # → 0.83

# Проверка орфографии
isv.spellcheck('prijatelj', 'isv')  # → True
isv.spellcheck('priyatel', 'isv')  # → False

# Процент корректных слов в тексте
isv.correctness('Dobry denj, kako jesi?', 'isv')  # → 1.00
isv.correctness('Dbory denj, kako jes?', 'isv')  # → 0.50

# Токенизация
isv.simple_tokenize('Dobry denj!')  # → ['dobry', 'denj']

# Доступные словари
isv.available_spellcheck_languages()  # → ['be', 'bg', 'cs', 'en', 'hr', 'isv', 'mk', 'pl', 'ru', 'sk', 'sl', 'sr', 'uk']

# Индекс качества текста (взвешенное среднее frequency, razumlivost, correctness)
isv.quality_index('Dobry denj, kako jesi?')  # → 0.81
isv.quality_index('Dobry denj, kako jesi?', frequency=0, razumlivost=0, correctness=1)  # → 1.00
isv.quality_index('črnogledniki slusajut izvěstoglašenje')  # → 0.22
```

✏️ Requirements
• Python 3.10+ (Optimized for 3.14)

✏️ License
This project is licensed under the MIT License.

✏️ Author
Mikhail Gorlatov - gorlatoff@gmail.com
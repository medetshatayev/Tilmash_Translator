# readability_indices.py

from nltk.tokenize import sent_tokenize, word_tokenize
import pyphen

def count_syllables(word, lang='en'):
    """
    Counts syllables in a given word, approximating for English with pyphen.
    For Russian/Kazakh, does a naive vowel-based approach.
    Adjust or improve this logic to match your language needs.
    """
    word = word.lower()

    if lang == 'ru':
        # Very naive approach for Russian
        vowels_ru = "аеёиоуыэюя"
        return max(1, sum(ch in vowels_ru for ch in word))
    elif lang == 'kk':
        # Simplified approach for Kazakh (may not be accurate)
        vowels_kk = "аеёиоуыэюяіүұө"
        return max(1, sum(ch in vowels_kk for ch in word))
    else:
        # Default: English, using pyphen
        dic = pyphen.Pyphen(lang='en')
        hyphens = dic.inserted(word)
        # Number of syllables ~ number of hyphens + 1
        return max(1, hyphens.count('-') + 1)


def flesch_reading_ease(text, lang='en'):
    """
    Flesch Reading Ease:
     - For English: 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
     - For Russian/Kazakh: sample adapted approach (not official).
    """
    sents = sent_tokenize(text, language='russian' if lang in ['ru', 'kk'] else 'english')
    words = [w for w in word_tokenize(text, language='russian' if lang in ['ru', 'kk'] else 'english') if w.isalpha()]
    num_sents = max(1, len(sents))
    num_words = max(1, len(words))
    syl_count = sum(count_syllables(w, lang) for w in words)

    asl = num_words / num_sents  # average sentence length
    asw = syl_count / num_words  # average syllables per word

    if lang == 'ru':
        # Sample adaptation for Russian
        return 206.835 - (1.3 * asl) - (60.1 * asw)
    elif lang == 'kk':
        # Simple adaptation for Kazakh
        return 206.835 - (1.2 * asl) - (70 * asw)
    else:
        # English default
        return 206.835 - (1.015 * asl) - (84.6 * asw)


def flesch_kincaid_grade_level(text, lang='en'):
    """
    Flesch-Kincaid Grade Level:
     - English: 0.39*(words/sentences) + 11.8*(syllables/words) - 15.59
     - Russian/Kazakh: naive adapted formula (not official).
    """
    sents = sent_tokenize(text, language='russian' if lang in ['ru', 'kk'] else 'english')
    words = [w for w in word_tokenize(text, language='russian' if lang in ['ru', 'kk'] else 'english') if w.isalpha()]
    num_sents = max(1, len(sents))
    num_words = max(1, len(words))
    syl_count = sum(count_syllables(w, lang) for w in words)

    asl = num_words / num_sents
    asw = syl_count / num_words

    if lang == 'ru':
        # Sample adaptation
        return (0.5 * asl) + (8.4 * asw) - 15.59
    elif lang == 'kk':
        # Simple adaptation
        return (0.5 * asl) + (9 * asw) - 13
    else:
        # English default
        return (0.39 * asl) + (11.8 * asw) - 15.59


def gunning_fog_index(text, lang='en'):
    """
    Gunning Fog Index:
     - 0.4 * [(words/sentences) + 100*(complex_words/words)]
     - 'Complex words' = >=3 syllables
    """
    sents = sent_tokenize(text, language='russian' if lang in ['ru', 'kk'] else 'english')
    words = [w for w in word_tokenize(text, language='russian' if lang in ['ru', 'kk'] else 'english') if w.isalpha()]
    num_sents = max(1, len(sents))
    num_words = max(1, len(words))
    complex_words = [w for w in words if count_syllables(w, lang) >= 3]

    perc_complex = (len(complex_words) / num_words) * 100
    asl = num_words / num_sents

    return 0.4 * (asl + perc_complex)


def smog_index(text, lang='en'):
    """
    SMOG Index:
     - 1.0430 * sqrt(30*(#complex_words / #sentences)) + 3.1291
     - 'Complex words' = >=3 syllables
     - Must have at least 3 sentences to be reliable
    """
    sents = sent_tokenize(text, language='russian' if lang in ['ru', 'kk'] else 'english')
    words = [w for w in word_tokenize(text, language='russian' if lang in ['ru', 'kk'] else 'english') if w.isalpha()]
    num_sents = len(sents)
    complex_words = [w for w in words if count_syllables(w, lang) >= 3]
    num_complex = len(complex_words)

    if num_sents < 3:
        return 0.0  # Not enough data for SMOG

    return 1.0430 * ((num_complex * (30 / num_sents)) ** 0.5) + 3.1291